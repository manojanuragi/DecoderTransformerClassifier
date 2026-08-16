
"""Client
  │
  ▼
API Gateway
  │
  ├──── /predict ─────► Inference
  │
  └──── /train ───────► TrainerClient
  """


import os
import uuid

import httpx

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from pydantic import (
    BaseModel,
    Field,
)


INFERENCE_URL = os.getenv(
    "INFERENCE_URL",
    "http://inference:8001",
)

TRAINER_URL = os.getenv(
    "TRAINER_URL",
    "http://trainer:8002",
)

MAX_TEXT_CHARS = int(
    os.getenv(
        "MAX_TEXT_CHARS",
        "10000",
    )
)


app = FastAPI(
    title="AG News Decoder API",
    version="1.0.0",
)


class PredictRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=10000,
    )


class TrainRequest(BaseModel):

    epochs: int = Field(
        default=3,
        ge=1,
        le=20,
    )

    batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
    )

    lr: float = Field(
        default=3e-4,
        gt=0,
        le=1e-1,
    )


class ThresholdsRequest(BaseModel):

    min_avg_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    max_low_confidence_rate: float = Field(default=0.40, ge=0.0, le=1.0)
    low_confidence_cutoff: float = Field(default=0.45, ge=0.0, le=1.0)
    min_samples: int = Field(default=50, ge=1, le=100000)
    window_size: int = Field(default=500, ge=10, le=100000)
    auto_retrain: bool = False


def _admin_ok(request: Request):
    expected = os.getenv("ADMIN_PASSWORD", "changeme")
    provided = request.headers.get("x-admin-password", "")
    return provided == expected


@app.middleware("http")
async def request_id(
    request: Request,
    call_next,
):

    request_id = request.headers.get(
        "x-request-id",
        str(uuid.uuid4()),
    )

    response = await call_next(
        request
    )

    response.headers[
        "x-request-id"
    ] = request_id

    return response


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "api",
    }


@app.post("/v1/predict")
async def predict(
    request: PredictRequest,
):

    headers = {

        "x-request-id":
            str(uuid.uuid4())
    }

    async with httpx.AsyncClient(
        timeout=30
    ) as client:

        try:

            response = await client.post(

                f"{INFERENCE_URL}"
                "/v1/predict",

                json=request.model_dump(),

                headers=headers,
            )

        except httpx.HTTPError as error:

            raise HTTPException(

                status_code=503,

                detail=(
                    f"inference unavailable: "
                    f"{error}"
                ),
            )

    if response.status_code >= 400:

        raise HTTPException(

            status_code=response.status_code,

            detail=response.text,
        )

    return response.json()


@app.post("/v1/train")
async def train(
    request: TrainRequest,
):

    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        try:

            response = await client.post(

                f"{TRAINER_URL}"
                "/v1/train",

                json=request.model_dump(),
            )

        except httpx.HTTPError as error:

            raise HTTPException(

                status_code=503,

                detail=(
                    f"trainer unavailable: "
                    f"{error}"
                ),
            )

    if response.status_code >= 400:

        raise HTTPException(

            status_code=response.status_code,

            detail=response.text,
        )

    return response.json()


@app.get("/v1/train/status")
async def train_status():

    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        response = await client.get(

            f"{TRAINER_URL}"
            "/v1/train/status"
        )

    return response.json()


@app.post("/v1/model/reload")
async def reload_model():

    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        response = await client.post(

            f"{INFERENCE_URL}"
            "/v1/reload"
        )

    if response.status_code >= 400:

        raise HTTPException(

            status_code=response.status_code,

            detail=response.text,
        )

    return response.json()


@app.get("/v1/metrics")
async def metrics_summary(http_request: Request):

    if not _admin_ok(http_request):
        raise HTTPException(status_code=401, detail="admin password required")

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{INFERENCE_URL}/v1/metrics",
            headers={
                "x-admin-password": http_request.headers.get(
                    "x-admin-password",
                    "",
                )
            },
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    return response.json()


@app.put("/v1/metrics/thresholds")
async def update_thresholds(
    request: ThresholdsRequest,
    http_request: Request,
):

    if not _admin_ok(http_request):
        raise HTTPException(status_code=401, detail="admin password required")

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.put(
            f"{INFERENCE_URL}/v1/metrics/thresholds",
            json=request.model_dump(),
            headers={
                "x-admin-password": http_request.headers.get(
                    "x-admin-password",
                    "",
                )
            },
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    return response.json()