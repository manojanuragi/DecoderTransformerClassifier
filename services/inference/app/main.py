import json
import os

from pathlib import Path

import torch

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from pydantic import (
    BaseModel,
    Field,
)

from tokenizers import Tokenizer

import sys

def _project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "shared").exists():
            return parent
    return Path("/app")


ROOT = _project_root()
for path in (ROOT, Path("/app")):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.model import (
    DecoderTransformerClassifier,
)

from shared.schema import (
    LABELS,
)

from shared.monitoring import MetricsMonitor


ARTIFACT_DIR = Path(
    os.getenv(
        "ARTIFACT_DIR",
        "/app/artifacts",
    )
)


MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "latest",
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


app = FastAPI(
    title="AG News Decoder Inference",
    version="1.0.0",
)


model = None
tokenizer = None
metadata = None
monitor = MetricsMonitor(ARTIFACT_DIR)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


def _admin_ok(request: Request):
    if not ADMIN_PASSWORD:
        return False
    return request.headers.get("x-admin-password", "") == ADMIN_PASSWORD

# type confirmation
class PredictRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=10000,
    )

# it laod models and other requirement
def load_artifacts():

    global model
    global tokenizer
    global metadata

    metadata_path = (
        ARTIFACT_DIR
        / "metadata.json"
    )

    tokenizer_path = (
        ARTIFACT_DIR
        / "tokenizer.json"
    )

    model_path = (
        ARTIFACT_DIR
        / "model.pt"
    )

    if not (
        metadata_path.exists()
        and tokenizer_path.exists()
        and model_path.exists()
    ):

        model = None
        tokenizer = None
        metadata = None

        return False

    metadata = json.loads(
        metadata_path.read_text()
    )

    tokenizer = (
        Tokenizer.from_file(
            str(tokenizer_path)
        )
    )

    config = metadata[
        "model_config"
    ]

    model = (
        DecoderTransformerClassifier(
            **config
        )
    )

    state = torch.load(

        model_path,

        map_location=DEVICE,

        weights_only=True,
    )

    model.load_state_dict(
        state
    )

    model.to(
        DEVICE
    )

    model.eval()

    return True


@app.on_event(
    "startup"
)
def startup():

    load_artifacts()


@app.get("/health")
def health():

    return {

        "status": "ok",

        "service":
            "inference",
    }


@app.get("/ready")
def ready():

    if model is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "model not trained"
            ),
        )

    return {

        "status": "ready",

        "device": DEVICE,

        "model_version":
            MODEL_VERSION,
    }


@app.post("/v1/reload")
def reload_model():

    if not load_artifacts():

        raise HTTPException(

            status_code=503,

            detail=(
                "model artifacts unavailable"
            ),
        )

    return {

        "status": "reloaded",

        "model_version":
            MODEL_VERSION,
    }


@app.post("/v1/predict")
def predict(
    request: PredictRequest,
):

    if (
        model is None
        or tokenizer is None
    ):

        if not load_artifacts():

            raise HTTPException(

                status_code=503,

                detail=(
                    "model not trained"
                ),
            )

    encoding = (
        tokenizer.encode(
            request.text
        )
    )

    ids = encoding.ids[

        :metadata[
            "model_config"
        ]["max_len"]

    ]

    if not ids:

        raise HTTPException(

            status_code=422,

            detail=(
                "text produced no tokens"
            ),
        )

    pad_id = metadata[
        "pad_id"
    ]

    max_len = metadata[
        "model_config"
    ]["max_len"]

    attention = [
        1
    ] * len(ids)

    ids += [
        pad_id
    ] * (
        max_len - len(ids)
    )

    attention += [
        0
    ] * (
        max_len - len(attention)
    )

    input_ids = torch.tensor(

        [ids],

        dtype=torch.long,

        device=DEVICE,
    )

    attention_mask = torch.tensor(

        [attention],

        dtype=torch.long,

        device=DEVICE,
    )

    with torch.inference_mode():

        logits = model(

            input_ids,

            attention_mask,
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )[0]
        )

        class_id = int(

            torch.argmax(
                probabilities
            ).item()
        )

    result = {

        "label":
            LABELS[class_id],

        "label_id":
            class_id,

        "confidence":
            round(
                float(
                    probabilities[
                        class_id
                    ].item()
                ),
                6,
            ),

        "probabilities": {

            LABELS[i]:
                round(
                    float(
                        probabilities[i]
                        .item()
                    ),
                    6,
                )

            for i in range(4)
        },

        "model_version":
            metadata.get(
                "model_version",
                MODEL_VERSION,
            ),
    }

    monitor.record_prediction(result)
    return result


class ThresholdsRequest(BaseModel):

    min_avg_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    max_low_confidence_rate: float = Field(default=0.40, ge=0.0, le=1.0)
    low_confidence_cutoff: float = Field(default=0.45, ge=0.0, le=1.0)
    min_samples: int = Field(default=50, ge=1, le=100000)
    window_size: int = Field(default=500, ge=10, le=100000)
    auto_retrain: bool = False


@app.get("/v1/metrics")
def metrics_summary(request: Request):
    if not _admin_ok(request):
        raise HTTPException(status_code=401, detail="admin password required")
    return monitor.summary()


@app.put("/v1/metrics/thresholds")
def update_thresholds(request: ThresholdsRequest, http_request: Request):
    if not _admin_ok(http_request):
        raise HTTPException(status_code=401, detail="admin password required")

    from shared.monitoring import MonitoringThresholds

    thresholds = MonitoringThresholds(**request.model_dump())
    monitor.save_thresholds(thresholds)
    summary = monitor.summary()
    return {
        "status": "updated",
        "thresholds": summary["thresholds"],
        "should_retrain": summary["should_retrain"],
        "retrain_reasons": summary["retrain_reasons"],
    }