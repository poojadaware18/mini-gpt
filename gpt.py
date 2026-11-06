"""
Mini GPT from scratch (character-level), following Karpathy's "Let's build GPT".
Run:  python gpt.py
Needs: pip install torch
Data:  download tiny shakespeare as input.txt
       https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Step 1: hyperparameters ----------
batch_size = 32        # sequences processed in parallel
block_size = 64        # context length
max_iters = 3000
eval_interval = 500
eval_iters = 100
learning_rate = 1e-3
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
# On CPU / low-end laptop use: batch_size=32, block_size=64, n_embd=128,
# n_head=4, n_layer=4, max_iters=3000
torch.manual_seed(1337)

# ---------- Step 2: data + tokenizer ----------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])  # next-char targets
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------- Step 3: self-attention ----------
class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # scaled scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # no peeking ahead
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


# ---------- Step 4: multi-head attention + feed-forward ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Step 5: transformer block (residual + layernorm) ----------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------- Step 6: full GPT model ----------
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ---------- Step 7: train ----------
model = GPTLanguageModel().to(device)
if __name__ == "__main__":
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss()
            print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---------- Step 8: save + generate text ----------
    torch.save(model.state_dict(), "gpt_char.pt")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))