# Build an LLM from Scratch

A hands-on implementation of a GPT-style large language model built from scratch in PyTorch, following the book *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Project Structure

```
├── ch02/   - Data loading and tokenization
├── ch03/   - Attention mechanisms
├── ch04/   - GPT model architecture
├── ch05/   - Pretraining and weight loading
└── ch06/   - Fine-tuning for classification
```

## Chapters

### Chapter 2 — Data Loading
- `dataloader.py` — sliding-window dataset and dataloader for text data

### Chapter 3 — Attention Mechanisms
- `selfAttention_v1/v2.py` — basic self-attention
- `CausalAttention.py` — masked causal self-attention
- `MultiHeadAttention.py` — multi-head attention
- `MultiHeadAttentionWrapper.py` — wrapper combining multiple attention heads

### Chapter 4 — GPT Model Architecture
- `GPTModel.py` — full GPT-2 style model (124M config)
- `TransformerBlock.py` — transformer block with attention + feed-forward
- `FeedForward.py` — position-wise feed-forward network
- `LayerNorm.py` — layer normalization
- `GELU.py` — GELU activation function
- `generate_text_simple.py` — greedy text generation

### Chapter 5 — Pretraining
- `pretraining_on_unlabeled_data.py` — training loop with loss tracking and text sampling
- `temperature_scaling.py` — temperature scaling and top-k sampling
- `gpt_download.py` — download pretrained OpenAI GPT-2 weights
- `load_pretrained_weights_from_openai.py` — load OpenAI weights into the model
- `load_and_save_model_weights.py` — save and reload model checkpoints

### Chapter 6 — Fine-tuning for Classification
- `prepare_dataset.py` — download and split the SMS spam dataset into train/validation/test CSVs

## Setup

```bash
pip install torch tiktoken
```

Chapter 6 additionally requires:
```bash
pip install pandas requests
```

## Model Configuration (GPT-2 124M)

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 50,257 |
| Context length | 256 |
| Embedding dim | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Dropout | 0.1 |
