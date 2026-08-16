"""Load model artifacts and classify news text."""

import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from shared.model import DecoderTransformerClassifier
from shared.schema import LABELS

try:
    from shared.monitoring import MetricsMonitor
except ImportError:
    MetricsMonitor = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Predictor:

    def __init__(self, artifact_dir=None):
        self.artifact_dir = Path(
            artifact_dir
            or os.getenv("ARTIFACT_DIR", "artifacts")
        )
        self.model = None
        self.tokenizer = None
        self.metadata = None
        self.model_version = os.getenv("MODEL_VERSION", "latest")
        self.monitor = (
            MetricsMonitor(self.artifact_dir)
            if MetricsMonitor is not None
            else None
        )

    @property
    def is_ready(self):
        return self.model is not None and self.tokenizer is not None

    def load(self):
        metadata_path = self.artifact_dir / "metadata.json"
        tokenizer_path = self.artifact_dir / "tokenizer.json"
        model_path = self.artifact_dir / "model.pt"

        if not (
            metadata_path.exists()
            and tokenizer_path.exists()
            and model_path.exists()
        ):
            self.model = None
            self.tokenizer = None
            self.metadata = None
            return False

        self.metadata = json.loads(metadata_path.read_text())
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        config = self.metadata["model_config"]
        self.model = DecoderTransformerClassifier(**config)

        state = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()
        return True

    def predict(self, text):
        if not self.is_ready and not self.load():
            raise RuntimeError(
                "Model not trained yet. Ask an admin to run training first."
            )

        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[: self.metadata["model_config"]["max_len"]]

        if not ids:
            raise ValueError("Text produced no tokens.")

        pad_id = self.metadata["pad_id"]
        max_len = self.metadata["model_config"]["max_len"]

        attention = [1] * len(ids)
        ids += [pad_id] * (max_len - len(ids))
        attention += [0] * (max_len - len(attention))

        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        attention_mask = torch.tensor(
            [attention],
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.inference_mode():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)[0]
            class_id = int(torch.argmax(probabilities).item())

        result = {
            "label": LABELS[class_id],
            "label_id": class_id,
            "confidence": round(float(probabilities[class_id].item()), 6),
            "probabilities": {
                LABELS[i]: round(float(probabilities[i].item()), 6)
                for i in range(len(LABELS))
            },
            "model_version": self.metadata.get(
                "model_version",
                self.model_version,
            ),
        }

        if self.monitor is not None:
            self.monitor.record_prediction(result)

        return result
