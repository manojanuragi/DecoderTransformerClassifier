import json
import os

from pathlib import Path

import torch

from fastapi import (
    FastAPI,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from tokenizers import Tokenizer

import sys

sys.path.insert(
    0,
    "/app",
)

from shared.model import (
    DecoderTransformerClassifier,
)

from shared.schema import (
    LABELS,
)


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

    return {

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