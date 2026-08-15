import os
import subprocess
import threading

from fastapi import (
    FastAPI,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)


app = FastAPI(
    title="AG News Trainer",
    version="1.0.0",
)


running = False

last_result = {
    "status": "idle"
}


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


def run_training(request):

    global running
    global last_result

    running = True

    last_result = {
        "status": "running"
    }

    command = [

        "python",

        "-m",

        "app.train",

        "--epochs",
        str(request.epochs),

        "--batch-size",
        str(request.batch_size),

        "--lr",
        str(request.lr),

        "--artifact-dir",

        os.getenv(
            "ARTIFACT_DIR",
            "/app/artifacts",
        ),
    ]

    try:

        process = subprocess.run(

            command,

            capture_output=True,

            text=True,

            timeout=86400,
        )

        last_result = {

            "status":
                (
                    "completed"
                    if process.returncode == 0
                    else "failed"
                ),

            "returncode":
                process.returncode,

            "stdout_tail":
                process.stdout[-4000:],

            "stderr_tail":
                process.stderr[-4000:],
        }

    except Exception as error:

        last_result = {

            "status": "failed",

            "error": str(error),
        }

    finally:

        running = False


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "trainer",
    }


@app.get("/ready")
def ready():

    return {
        "status": "ready"
    }


@app.post("/v1/train")
def start_training(
    request: TrainRequest,
):

    global running

    if running:

        raise HTTPException(

            status_code=409,

            detail=(
                "training already running"
            ),
        )

    thread = threading.Thread(

        target=run_training,

        args=(request,),

        daemon=True,
    )

    thread.start()

    return {
        "status": "accepted"
    }


@app.get("/v1/train/status")
def training_status():

    return {

        "running": running,

        **last_result,
    }