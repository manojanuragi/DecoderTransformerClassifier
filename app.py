"""
Gradio frontend for the Decoder Transformer Classifier.

Standalone mode (Hugging Face Spaces): loads the model locally and runs training
in-process. Set ADMIN_PASSWORD as a Space secret.

API mode (Docker Compose): proxies requests to the FastAPI gateway. Set API_URL,
for example http://api:8000.
"""

import io
import os
import sys
import threading
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import gradio as gr
import httpx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.predict import Predictor
from shared.monitoring import MetricsMonitor, MonitoringThresholds

API_URL = os.getenv("API_URL", "").rstrip("/")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

predictor = Predictor(ARTIFACT_DIR)
monitor = MetricsMonitor(ARTIFACT_DIR)
training_state = {
    "running": False,
    "status": "idle",
    "log": "",
}


def use_api():
    return bool(API_URL)


def format_probabilities(probabilities):
    lines = []
    for label, score in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"{label}: {score:.2%}")
    return "\n".join(lines)


def format_metrics_summary(summary):
    lines = [
        f"predictions in window: {summary['prediction_count']} / {summary['window_size']}",
        f"average confidence: {summary['avg_confidence']}",
        f"low-confidence rate: {summary['low_confidence_rate']}",
        f"should retrain: {summary['should_retrain']}",
        "retrain reasons:",
    ]
    for reason in summary.get("retrain_reasons", []):
        lines.append(f"  - {reason}")

    baseline = summary.get("baseline") or {}
    if baseline:
        lines.append("")
        lines.append(
            "training baseline: "
            f"macro_f1={baseline.get('macro_f1')}, "
            f"accuracy={baseline.get('accuracy')}"
        )

    class_counts = summary.get("class_counts") or {}
    if class_counts:
        lines.append("")
        lines.append("recent label counts:")
        for label, count in sorted(class_counts.items()):
            lines.append(f"  {label}: {count}")

    thresholds = summary.get("thresholds") or {}
    if thresholds:
        lines.append("")
        lines.append("thresholds:")
        lines.append(
            f"  min_avg_confidence={thresholds.get('min_avg_confidence')}"
        )
        lines.append(
            f"  max_low_confidence_rate={thresholds.get('max_low_confidence_rate')}"
        )
        lines.append(
            f"  low_confidence_cutoff={thresholds.get('low_confidence_cutoff')}"
        )
        lines.append(f"  min_samples={thresholds.get('min_samples')}")
        lines.append(f"  auto_retrain={thresholds.get('auto_retrain')}")

    return "\n".join(lines)


def fetch_metrics(password):
    if password != ADMIN_PASSWORD:
        return "Wrong admin password."

    try:
        if use_api():
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"{API_URL}/v1/metrics",
                    headers={"x-admin-password": password},
                )
            if response.status_code >= 400:
                return f"Could not load metrics: {response.text}"
            summary = response.json()
        else:
            summary = monitor.summary()

        return format_metrics_summary(summary)

    except Exception as error:
        return f"Could not load metrics: {error}"


def save_thresholds(
    password,
    min_avg_confidence,
    max_low_confidence_rate,
    low_confidence_cutoff,
    min_samples,
    window_size,
    auto_retrain,
):
    if password != ADMIN_PASSWORD:
        return "Wrong admin password.", fetch_metrics(password)

    payload = {
        "min_avg_confidence": float(min_avg_confidence),
        "max_low_confidence_rate": float(max_low_confidence_rate),
        "low_confidence_cutoff": float(low_confidence_cutoff),
        "min_samples": int(min_samples),
        "window_size": int(window_size),
        "auto_retrain": bool(auto_retrain),
    }

    try:
        if use_api():
            with httpx.Client(timeout=30) as client:
                response = client.put(
                    f"{API_URL}/v1/metrics/thresholds",
                    json=payload,
                    headers={"x-admin-password": password},
                )
            if response.status_code >= 400:
                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                return f"Could not save thresholds: {detail}", fetch_metrics(password)
        else:
            monitor.save_thresholds(MonitoringThresholds(**payload))

        return "Monitoring thresholds saved.", fetch_metrics(password)

    except Exception as error:
        return f"Could not save thresholds: {error}", fetch_metrics(password)


def maybe_auto_retrain(summary):
    thresholds = summary.get("thresholds") or {}
    if not thresholds.get("auto_retrain"):
        return None
    if not summary.get("should_retrain"):
        return None
    if training_state["running"]:
        return "Retrain threshold hit, but training is already running."

    message, _ = start_training(
        ADMIN_PASSWORD,
        1,
        64,
        0.0003,
    )
    return f"Auto-retrain triggered: {message}"


def classify_text(text):
    text = (text or "").strip()
    if not text:
        return "Enter some news text first.", ""

    auto_message = None
    try:
        if use_api():
            with httpx.Client(timeout=60) as client:
                response = client.post(
                    f"{API_URL}/v1/predict",
                    json={"text": text},
                )
            if response.status_code >= 400:
                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                return f"Prediction failed: {detail}", ""
            result = response.json()
        else:
            result = predictor.predict(text)
            auto_message = maybe_auto_retrain(monitor.summary())

        headline = (
            f"**{result['label']}** "
            f"({result['confidence']:.1%} confidence)"
        )
        probs = format_probabilities(result["probabilities"])
        if not use_api() and auto_message:
            headline += f"\n\n_{auto_message}_"
        return headline, probs

    except Exception as error:
        return f"Prediction failed: {error}", ""


def verify_admin(password):
    if password == ADMIN_PASSWORD:
        return (
            gr.update(visible=True),
            "Admin access granted.",
        )
    return (
        gr.update(visible=False),
        "Wrong password.",
    )


def _run_local_training(epochs, batch_size, lr):
    global training_state

    training_state["running"] = True
    training_state["status"] = "running"
    training_state["log"] = "Training started...\n"

    buffer = io.StringIO()

    try:
        from services.trainer.app.train import train

        args = Namespace(
            artifact_dir=ARTIFACT_DIR,
            epochs=int(epochs),
            batch_size=int(batch_size),
            eval_batch_size=min(int(batch_size) * 2, 128),
            max_len=int(os.getenv("MAX_LEN", 128)),
            vocab_size=int(os.getenv("TOKENIZER_VOCAB_SIZE", 16000)),
            d_model=int(os.getenv("D_MODEL", 256)),
            n_heads=int(os.getenv("N_HEADS", 8)),
            n_layers=int(os.getenv("N_LAYERS", 6)),
            d_ff=int(os.getenv("D_FF", 1024)),
            dropout=float(os.getenv("DROPOUT", 0.1)),
            lr=float(lr),
            weight_decay=0.01,
            workers=0,
            seed=42,
            model_version=os.getenv("MODEL_VERSION", "latest"),
            rebuild_tokenizer=False,
        )

        with redirect_stdout(buffer):
            train(args)

        predictor.load()
        training_state["status"] = "completed"
        training_state["log"] += buffer.getvalue()
        training_state["log"] += "\nTraining finished. Model reloaded.\n"

    except Exception as error:
        training_state["status"] = "failed"
        training_state["log"] += buffer.getvalue()
        training_state["log"] += f"\nTraining failed: {error}\n"

    finally:
        training_state["running"] = False


def start_training(password, epochs, batch_size, lr):
    if password != ADMIN_PASSWORD:
        return "Wrong admin password.", training_status()

    if training_state["running"]:
        return "Training is already running.", training_status()

    if use_api():
        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{API_URL}/v1/train",
                    json={
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "lr": float(lr),
                    },
                )
            if response.status_code >= 400:
                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                return f"Could not start training: {detail}", training_status()

            training_state["running"] = True
            training_state["status"] = "running"
            training_state["log"] = "Training accepted by API.\n"
            return "Training started.", training_status()

        except Exception as error:
            return f"Could not reach API: {error}", training_status()

    thread = threading.Thread(
        target=_run_local_training,
        args=(epochs, batch_size, lr),
        daemon=True,
    )
    thread.start()
    return "Training started.", training_status()


def training_status():
    if use_api() and training_state["running"]:
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(f"{API_URL}/v1/train/status")
            if response.status_code < 400:
                payload = response.json()
                running = payload.get("running", False)
                training_state["running"] = running
                training_state["status"] = payload.get("status", "unknown")

                chunks = [
                    f"running: {running}",
                    f"status: {training_state['status']}",
                ]
                if payload.get("stdout_tail"):
                    chunks.append("\n--- stdout ---")
                    chunks.append(payload["stdout_tail"])
                if payload.get("stderr_tail"):
                    chunks.append("\n--- stderr ---")
                    chunks.append(payload["stderr_tail"])
                if payload.get("error"):
                    chunks.append(f"\nerror: {payload['error']}")

                if not running and training_state["status"] == "completed":
                    with httpx.Client(timeout=30) as client:
                        client.post(f"{API_URL}/v1/model/reload")

                return "\n".join(chunks)
        except Exception as error:
            return training_state["log"] + f"\nStatus check failed: {error}"

    lines = [
        f"running: {training_state['running']}",
        f"status: {training_state['status']}",
        "",
        training_state["log"] or "No training logs yet.",
    ]
    return "\n".join(lines)


def reload_model(password):
    if password != ADMIN_PASSWORD:
        return "Wrong admin password."

    try:
        if use_api():
            with httpx.Client(timeout=30) as client:
                response = client.post(f"{API_URL}/v1/model/reload")
            if response.status_code >= 400:
                return f"Reload failed: {response.text}"
            return "Model reloaded from artifacts."

        if predictor.load():
            return "Model reloaded from artifacts."
        return "No trained artifacts found yet."

    except Exception as error:
        return f"Reload failed: {error}"


def build_ui():
    mode = "API proxy" if use_api() else "standalone"
    subtitle = (
        f"Classify AG News headlines with a decoder-only Transformer ({mode} mode)."
    )

    with gr.Blocks(title="AG News Decoder Classifier") as demo:
        gr.Markdown("# AG News Decoder Classifier")
        gr.Markdown(subtitle)

        with gr.Tab("Classify"):
            text_input = gr.Textbox(
                label="News text",
                placeholder="Paste a headline or short article...",
                lines=4,
            )
            classify_btn = gr.Button("Classify", variant="primary")
            result_md = gr.Markdown()
            probs_box = gr.Textbox(
                label="Class probabilities",
                lines=4,
                interactive=False,
            )

            classify_btn.click(
                classify_text,
                inputs=text_input,
                outputs=[result_md, probs_box],
            )
            text_input.submit(
                classify_text,
                inputs=text_input,
                outputs=[result_md, probs_box],
            )

        with gr.Tab("Admin"):
            gr.Markdown(
                "Training is restricted to admins. "
                "Set `ADMIN_PASSWORD` in your environment or Hugging Face Space secrets."
            )

            admin_password = gr.Textbox(
                label="Admin password",
                type="password",
            )
            login_btn = gr.Button("Unlock admin panel")
            login_msg = gr.Markdown()

            with gr.Column(visible=False) as admin_panel:
                epochs = gr.Slider(
                    1,
                    10,
                    value=1,
                    step=1,
                    label="Epochs",
                )
                batch_size = gr.Slider(
                    16,
                    256,
                    value=64,
                    step=16,
                    label="Batch size",
                )
                lr = gr.Number(
                    value=0.0003,
                    label="Learning rate",
                )
                train_btn = gr.Button("Start training", variant="primary")
                refresh_btn = gr.Button("Refresh training status")
                reload_btn = gr.Button("Reload model")
                train_msg = gr.Markdown()
                status_box = gr.Textbox(
                    label="Training status",
                    lines=14,
                    interactive=False,
                )

                gr.Markdown("### Monitoring")
                metrics_box = gr.Textbox(
                    label="Live monitoring summary",
                    lines=16,
                    interactive=False,
                )
                metrics_refresh_btn = gr.Button("Refresh metrics")

                min_avg_confidence = gr.Slider(
                    0.1,
                    0.95,
                    value=0.55,
                    step=0.01,
                    label="Retrain if avg confidence below",
                )
                max_low_confidence_rate = gr.Slider(
                    0.05,
                    0.95,
                    value=0.40,
                    step=0.01,
                    label="Retrain if low-confidence rate above",
                )
                low_confidence_cutoff = gr.Slider(
                    0.1,
                    0.9,
                    value=0.45,
                    step=0.01,
                    label="Low-confidence cutoff",
                )
                min_samples = gr.Slider(
                    10,
                    1000,
                    value=50,
                    step=10,
                    label="Minimum predictions before checking",
                )
                window_size = gr.Slider(
                    50,
                    5000,
                    value=500,
                    step=50,
                    label="Monitoring window size",
                )
                auto_retrain = gr.Checkbox(
                    label="Auto-retrain when thresholds are breached",
                    value=False,
                )
                save_thresholds_btn = gr.Button("Save monitoring thresholds")
                metrics_msg = gr.Markdown()

            login_btn.click(
                verify_admin,
                inputs=admin_password,
                outputs=[admin_panel, login_msg],
            )

            train_btn.click(
                start_training,
                inputs=[admin_password, epochs, batch_size, lr],
                outputs=[train_msg, status_box],
            )
            refresh_btn.click(
                training_status,
                outputs=status_box,
            )
            reload_btn.click(
                reload_model,
                inputs=admin_password,
                outputs=train_msg,
            )
            metrics_refresh_btn.click(
                fetch_metrics,
                inputs=admin_password,
                outputs=metrics_box,
            )
            save_thresholds_btn.click(
                save_thresholds,
                inputs=[
                    admin_password,
                    min_avg_confidence,
                    max_low_confidence_rate,
                    low_confidence_cutoff,
                    min_samples,
                    window_size,
                    auto_retrain,
                ],
                outputs=[metrics_msg, metrics_box],
            )

        gr.Markdown(
            "Labels: World, Sports, Business, Sci/Tech. "
            "On Hugging Face Spaces, use 1 epoch first because free CPU hardware is slow."
        )

    return demo


if __name__ == "__main__":
    if not use_api():
        predictor.load()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
