# Mini GPT from Scratch

A character-level GPT (decoder-only Transformer) built from scratch in PyTorch, following Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out". Trained on the Tiny Shakespeare dataset to generate Shakespeare-style text, with a Streamlit web app to try it out.

**Live demo:** _add your Streamlit link here after deploying_

![App screenshot](screenshot.png)

## Features
- GPT model written from scratch (no pre-built Transformer classes)
- Trains on CPU in about 35-40 minutes
- Command-line text generation (`generate.py`)
- Streamlit web app with prompt, length and temperature controls (`app.py`)

## Architecture
- Token embeddings + learned positional embeddings
- Multi-head causal self-attention (scaled dot-product, masked so tokens cannot see the future)
- Position-wise feed-forward network (4x expansion)
- Residual connections and LayerNorm in every block
- Final LayerNorm + linear head over the character vocabulary
- Trained with cross-entropy loss and the AdamW optimizer

**Model size:** 0.82M parameters (4 layers, 4 heads, 128-dim embeddings, context length 64)

## Project structure
```
mini-gpt/
├── gpt.py             # model, training loop (run directly to train)
├── generate.py        # generate text in the terminal from the trained model
├── app.py             # Streamlit web app
├── download.py        # downloads the Tiny Shakespeare dataset
├── input.txt          # dataset
├── gpt_char.pt        # trained model weights
├── requirements.txt
└── README.md
```

## How to run

### 1. Setup
```bash
pip install -r requirements.txt
python download.py
```

### 2. Train (optional, trained weights are included)
```bash
python gpt.py
```
Training takes roughly 35-40 minutes on a CPU and saves `gpt_char.pt`. For better results on a GPU, increase `n_embd`, `n_layer`, `block_size` and `max_iters` at the top of `gpt.py`.

### 3. Generate text in the terminal
```bash
python generate.py
```

### 4. Run the web app
```bash
python -m streamlit run app.py
```
Then open http://localhost:8501.

## Results (CPU, 3000 steps)

| Step | Train loss | Val loss |
|------|-----------|----------|
| 0    | 4.3354    | 4.3353   |
| 500  | 1.9691    | 2.0315   |
| 1000 | 1.7168    | 1.8710   |
| 1500 | 1.6137    | 1.7725   |
| 2000 | 1.5477    | 1.7168   |
| 2500 | 1.5139    | 1.7094   |
| 2999 | 1.4822    | 1.6902   |

Sample generated text (prompt: `ROMEO:`). The model learned play structure, speaker names and Shakespeare-like word shapes, but it is a very small model, so many words are made up:

```
ROMEO:
No, proad him the unpresent see the day's duck and daughter match,
Whithou make unto see mether, of the corrow,
Which no be let time preserve the Murdents and man:
Sir, Were it words be in all your fearful.

ANTIGONUS:
How you mark the cousin!
```

## What I learned
- How self-attention works: queries, keys and values, scaling, and causal masking.
- Why residual connections and LayerNorm make deeper Transformers trainable.
- How to build a training pipeline: batching, train/validation split, loss estimation.
- How loss relates to output quality, and how model size and context length affect results.
- How to turn a trained model into a small web app.

## Future improvements
- Replace the character-level tokenizer with BPE (e.g. `tiktoken`)
- Train a larger model on a GPU
- Train on a custom dataset such as Hindi or Marathi text

## Credits
Based on Andrej Karpathy's "Let's build GPT" lecture and the Tiny Shakespeare dataset.
