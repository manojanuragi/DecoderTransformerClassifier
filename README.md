# Decoder Transformer Classifier

This project trains a decoder-only Transformer to classify news headlines into one of four AG News categories: World, Sports, Business, and Sci/Tech. You can use it through a Gradio web UI, a REST API, or deploy it as a Hugging Face Space.

If you have seen BERT-style encoders before, the difference here is mostly architectural. We use a causal decoder-only stack (the same family of design as GPT) and adapt it for classification rather than next-token prediction.


Why a decoder for classification?

Text classification often appears in courses with bag-of-words models or fine-tuned BERT. Both are valid, but they teach different things. Here we take the decoder block from a language model and use it as a sequence encoder.

Self-attention in brief: a Transformer does not read text strictly left-to-right like an RNN. Each token gets a query, key, and value vector. Attention scores decide how much every other token should influence the current one. Stack enough layers and the model builds up from local word patterns to phrases and eventually broader topic cues. For a headline like "Apple unveils new AI chip for data centers", early layers might pick up entity names and verbs; deeper layers combine those into something closer to a Sci/Tech product launch representation.

Encoders (BERT) use bidirectional attention, meaning each token can look at the full sequence at once. That works well when the entire input is available upfront. Decoders use causal masked self-attention: token i can only attend to positions at or before i. That constraint is what makes autoregressive language modeling work, since the model never sees future words during training.

Using a decoder for classification is less common than fine-tuning BERT, but it is a useful exercise. You keep the causal inductive bias and still end up with a fixed-length vector at the end of the sequence for the classifier head.

How text becomes a label: first, a BPE tokenizer trained on AG News splits the input. Token IDs are embedded and combined with learned positional embeddings, scaled by sqrt(d_model) as in the original Transformer paper. Each decoder block applies pre-norm LayerNorm, causal multi-head self-attention, a residual, another LayerNorm, a two-layer feed-forward network with GELU, and a second residual. Instead of a dedicated CLS token, we pool the hidden state at the last non-padded position. In a left-to-right decoder that position has seen the whole sequence under the causal mask, so it works as a summary vector. A linear layer maps that vector to four logits; softmax at inference gives class probabilities. Padding tokens are masked out in attention so they do not leak into other positions.

Default model size: max_len 128, d_model 256, n_heads 8, n_layers 6, d_ff 1024, dropout 0.1.

Training uses cross-entropy loss on the AG News labels. Optimization is AdamW with weight decay and gradient clipping. Checkpoints are saved when validation macro-F1 improves on the test split, so we keep the best generalizing weights rather than whatever the last epoch produced.


Evaluation report

The figures below come from the latest completed training job on CPU: 1 epoch, batch size 64, learning rate 0.0003. Evaluation is the official AG News test split (7,600 headlines). The same accuracy and macro-F1 are written to artifacts/metadata.json and used as the monitoring baseline.

Setup:

  dataset     AG News (120,000 train / 7,600 test)
  tokenizer   BPE, vocab_size 16000
  model       max_len 128, d_model 256, n_heads 8, n_layers 6, d_ff 1024, dropout 0.1
  optimizer   AdamW, lr 0.0003, batch_size 64, gradient clip 1.0
  device      cpu
  epochs      1

Test set:

  train loss  0.3192
  accuracy    0.9228
  macro-F1    0.9228

Sample prediction after POST /v1/model/reload:

  input: "Stock markets rally on strong earnings report"
  label: Business (confidence 0.987)

Re-running training with more epochs will overwrite these numbers. Poll GET /v1/train/status or open artifacts/metadata.json for the current checkpoint.


System design

Training and inference have different needs. Training pulls the Hugging Face dataset, builds a tokenizer, and runs for a long time. Inference only loads weights and scores a single string. Splitting them keeps the serving path light.

Services when running with Docker Compose:

  gradio      7860   browser UI (Classify tab public, Admin tab password-protected)
  api         8000   REST gateway
  inference   8001   model loading and prediction
  trainer     8002   background training jobs

The api service is the single entry point for curl and programmatic clients. POST /v1/predict goes to inference. POST /v1/train goes to trainer, which writes artifacts to disk. After training finishes, call POST /v1/model/reload so inference picks up the new weights without a full restart.

The Gradio app on port 7860 talks to the api service in Docker mode. The Classify tab is public. The Admin tab requires ADMIN_PASSWORD and can start training, reload the model, view monitoring metrics, and edit retrain thresholds.


Getting started

You need Docker and Docker Compose. Python 3.11 and PyTorch are enough to run the unit tests locally without containers.

Start everything:

  docker compose up --build -d

Open the Gradio UI at http://localhost:7860

Create a local .env file from .env.example and set ADMIN_PASSWORD before starting admin features:

  cp .env.example .env

Sanity check:

  curl http://localhost:8000/health
  curl http://localhost:8001/ready
  curl http://localhost:7860

Train (runs in the background; poll status until status is completed):

  curl -X POST http://localhost:8000/v1/train \
    -H "Content-Type: application/json" \
    -d '{"epochs": 1, "batch_size": 64, "lr": 0.0003}'

Training writes tokenizer.json, model.pt, and metadata.json into artifacts/. On CPU this can take a while; use epochs 1 first if you just want to verify the pipeline.

Predict:

  curl -X POST http://localhost:8000/v1/model/reload

  curl -X POST http://localhost:8000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Stock markets rally on strong earnings report"}'


Gradio UI

Classify tab: paste a headline, get a label and class probabilities.

Admin tab: unlock with the admin password configured in your environment, then you can:

  start or monitor training
  reload the model from artifacts
  refresh monitoring metrics (admin only)
  set retrain thresholds and optional auto-retrain

Run Gradio locally without Docker (standalone mode, no microservices):

  pip install -r requirements.txt
  python app.py

Set ADMIN_PASSWORD in the environment before using the Admin tab.


API

GET  /health                    liveness check
POST /v1/predict                classify input text
POST /v1/train                  start a training job
GET  /v1/train/status           poll training progress
POST /v1/model/reload           hot-reload artifacts into inference
GET  /v1/metrics                monitoring summary (admin only)
PUT  /v1/metrics/thresholds     update retrain thresholds (admin only)

Admin endpoints require the x-admin-password request header matching the configured ADMIN_PASSWORD.


Monitoring and retrain thresholds

Every prediction is logged to artifacts/prediction_log.jsonl. The service keeps a rolling window and computes average confidence, low-confidence rate, and recent label counts. Thresholds live in artifacts/monitoring_config.json and can also be seeded from environment variables.

Metrics are admin-only. Public users can classify text but cannot read monitoring data.

Default retrain rules once at least 50 predictions are in the window:

  retrain if average confidence drops below 0.55
  retrain if more than 40% of predictions are below 0.45 confidence
  retrain if confidence looks too weak compared with the saved training macro-F1 baseline

Environment variables:

  ADMIN_PASSWORD                required for admin UI and protected API routes
  METRICS_MIN_AVG_CONFIDENCE=0.55
  METRICS_MAX_LOW_CONFIDENCE_RATE=0.40
  METRICS_LOW_CONFIDENCE_CUTOFF=0.45
  METRICS_MIN_SAMPLES=50
  METRICS_WINDOW=500
  METRICS_F1_DROP=0.05
  AUTO_RETRAIN=false

Check metrics as admin:

  curl http://localhost:8000/v1/metrics \
    -H "x-admin-password: <admin-password>"

Update thresholds as admin:

  curl -X PUT http://localhost:8000/v1/metrics/thresholds \
    -H "Content-Type: application/json" \
    -H "x-admin-password: <admin-password>" \
    -d '{"min_avg_confidence":0.55,"max_low_confidence_rate":0.40,"low_confidence_cutoff":0.45,"min_samples":50,"window_size":500,"auto_retrain":false}'


Deploy on Hugging Face Spaces

This repo includes app.py and requirements.txt at the project root for a Gradio Space.

1. Go to https://huggingface.co/new-space
2. Pick Gradio as the SDK and connect this repository
3. Open Space Settings -> Repository secrets and add ADMIN_PASSWORD as a secret (do not commit it to the repo)
4. Copy the YAML header from README_SPACE.md into the Space README if prompted

On Spaces the app runs in standalone mode: it loads and trains the model directly without the Docker microservices. Start with 1 epoch on the Admin tab because free CPU hardware is slow. Keep AUTO_RETRAIN off on Spaces unless you accept long training runs.


Project layout

  shared/model.py          DecoderTransformerClassifier
  shared/schema.py         label map and ModelConfig
  shared/predict.py        load artifacts and run inference
  shared/monitoring.py     prediction logging, metrics, retrain checks
  app.py                   Gradio frontend
  requirements.txt         dependencies for Gradio / Hugging Face Spaces
  Dockerfile.gradio        Gradio container for docker compose
  services/api             FastAPI gateway
  services/inference       prediction service
  services/trainer         training service and train.py
  artifacts/               model outputs (gitignored except .gitkeep)
  tests/                   unit tests

Run tests without Docker:

  pip install torch pytest
  pytest tests/test-model.py -v


References

Vaswani et al., Attention Is All You Need (2017) — https://arxiv.org/abs/1706.03762
AG News on Hugging Face — https://huggingface.co/datasets/ag_news
Radford et al., GPT series — decoder-only stack with causal masking
