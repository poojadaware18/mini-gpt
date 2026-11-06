import streamlit as st
import torch
import torch.nn.functional as F

from gpt import model, encode, decode, device, block_size

st.set_page_config(page_title="Mini GPT", page_icon="🎭")


@st.cache_resource
def load_model():
    model.load_state_dict(torch.load("gpt_char.pt", map_location=device))
    model.eval()
    return model


@torch.no_grad()
def generate(m, prompt, max_new_tokens, temperature):
    if prompt:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = m(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return decode(idx[0].tolist())


st.title("🎭 Mini GPT")
st.caption(
    "A character-level GPT (0.82M parameters) built from scratch in PyTorch "
    "and trained on Shakespeare. It is a small educational model, so expect "
    "Shakespeare-style text with many made-up words."
)

m = load_model()

prompt = st.text_input("Starting text (optional)", value="ROMEO:")
col1, col2 = st.columns(2)
length = col1.slider("Characters to generate", 100, 1000, 400, step=50)
temperature = col2.slider("Temperature (higher = more random)", 0.3, 1.5, 0.8, step=0.1)

if st.button("Generate", type="primary"):
    try:
        with st.spinner("Generating..."):
            text = generate(m, prompt, length, temperature)
        st.code(text, language=None)
    except KeyError:
        st.error("Use only characters found in the Shakespeare text (English letters and basic punctuation).")