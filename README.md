# Decoder Transformer Classifier

This project trains a decoder-only Transformer to classify news headlines into one of four AG News categories: World, Sports, Business, and Sci/Tech. The model sits behind a small three-service API where you can train it, reload checkpoints, and run predictions over HTTP.

If you have seen BERT-style encoders before, the difference here is mostly architectural. We use a causal decoder-only stack (the same family of design as GPT) and adapt it for classification rather than next-token prediction.


Why a decoder for classification?

Text classification often appears in courses with bag-of-words models or fine-tuned BERT. Both are valid, but they teach different things. Here we take the decoder block from a language model and use it as a sequence encoder.

Self-attention in brief: a Transformer does not read text strictly left-to-right like an RNN. Each token gets a query, key, and value vector. Attention scores decide how much every other token should influence the current one. Stack enough layers and the model builds up from local word patterns to phrases and eventually broader topic cues. For a headline like "Apple unveils new AI chip for data centers", early layers might pick up entity names and verbs; deeper layers combine those into something closer to a Sci/Tech product launch representation.

Encoders (BERT) use bidirectional attention, meaning each token can look at the full sequence at once. That works well when the entire input is available upfront. Decoders use causal masked self-attention: token i can only attend to positions at or before i. That constraint is what makes autoregressive language modeling work, since the model never sees future words during training.

Using a decoder for classification is less common than fine-tuning BERT, but it is a useful exercise. You keep the causal inductive bias and still end up with a fixed-length vector at the end of the sequence for the classifier head.

How text becomes a label: first, a BPE tokenizer trained on AG News splits the input. Token IDs are embedded and combined with learned positional embeddings, scaled by sqrt(d_model) as in the original Transformer paper. Each decoder block applies pre-norm LayerNorm, causal multi-head self-attention, a residual, another LayerNorm, a two-layer feed-forward network with GELU, and a second residual. Instead of a dedicated CLS token, we pool the hidden state at the last non-padded position. In a left-to-right decoder that position has seen the whole sequence under the causal mask, so it works as a summary vector. A linear layer maps that vector to four logits; softmax at inference gives class probabilities. Padding tokens are masked out in attention so they do not leak into other positions.

Default model size: max_len 128, d_model 256, n_heads 8, n_layers 6, d_ff 1024, dropout 0.1.

Training uses cross-entropy loss on the AG News labels. Optimization is AdamW with weight decay and gradient clipping. Checkpoints are saved when validation macro-F1 improves on the test split, so we keep the best generalizing weights rather than whatever the last epoch produced.


System design

Training and inference have different needs. Training pulls the Hugging Face dataset, builds a tokenizer, and runs for a long time. Inference only loads weights and scores a single string. Splitting them keeps the serving path light.

The api service on port 8000 is the single entry point. POST /v1/predict goes to inference on 8001. POST /v1/train goes to trainer on 8002, which runs training in a background thread and writes artifacts to disk. After training finishes, call POST /v1/model/reload so inference picks up the new weights without a full restart.


Getting started

You need Docker and Docker Compose. Python 3.11 and PyTorch are enough to run the unit tests locally without containers.

Start everything:

  docker compose up --build -d

Sanity check:

  curl http://localhost:8000/health
  curl http://localhost:8001/health
  curl http://localhost:8000/v1/train/status

Train (runs in the background; poll status until status is completed):

  curl -X POST http://localhost:8000/v1/train \
    -H "Content-Type: application/json" \
    -d '{"epochs": 3, "batch_size": 64, "lr": 0.0003}'

Training writes tokenizer.json, model.pt, and metadata.json into artifacts/. On CPU this can take a while; use epochs 1 first if you just want to verify the pipeline.

Predict:

  curl -X POST http://localhost:8000/v1/model/reload

  curl -X POST http://localhost:8000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Stock markets rally on strong earnings report"}'


API

GET  /health              liveness check
POST /v1/predict          classify input text
POST /v1/train            start a training job
GET  /v1/train/status     poll training progress
POST /v1/model/reload     hot-reload artifacts into inference


Project layout

shared/model.py and shared/schema.py hold the model and config. services/api is the gateway, services/inference serves predictions, services/trainer runs training via train.py. artifacts/ stores model outputs (gitignored except .gitkeep). tests/ has shape and forward-pass checks.

Run tests without Docker:

  pip install torch pytest
  pytest tests/test-model.py -v


References

Vaswani et al., Attention Is All You Need (2017) — https://arxiv.org/abs/1706.03762
AG News on Hugging Face — https://huggingface.co/datasets/ag_news
Radford et al., GPT series — decoder-only stack with causal masking
