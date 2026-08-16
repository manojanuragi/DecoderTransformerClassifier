"""Prediction monitoring, rolling metrics, and retrain thresholds."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MonitoringThresholds:
    min_avg_confidence: float = 0.55
    max_low_confidence_rate: float = 0.40
    low_confidence_cutoff: float = 0.45
    min_samples: int = 50
    window_size: int = 500
    auto_retrain: bool = False


class MetricsMonitor:

    def __init__(self, artifact_dir=None):
        self.artifact_dir = Path(
            artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts")
        )
        self.log_path = self.artifact_dir / "prediction_log.jsonl"
        self.config_path = self.artifact_dir / "monitoring_config.json"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def load_thresholds(self):
        if not self.config_path.exists():
            thresholds = MonitoringThresholds(
                min_avg_confidence=float(
                    os.getenv("METRICS_MIN_AVG_CONFIDENCE", 0.55)
                ),
                max_low_confidence_rate=float(
                    os.getenv("METRICS_MAX_LOW_CONFIDENCE_RATE", 0.40)
                ),
                low_confidence_cutoff=float(
                    os.getenv("METRICS_LOW_CONFIDENCE_CUTOFF", 0.45)
                ),
                min_samples=int(os.getenv("METRICS_MIN_SAMPLES", 50)),
                window_size=int(os.getenv("METRICS_WINDOW", 500)),
                auto_retrain=os.getenv("AUTO_RETRAIN", "").lower() in {
                    "1",
                    "true",
                    "yes",
                },
            )
            self.save_thresholds(thresholds)
            return thresholds

        data = json.loads(self.config_path.read_text())
        return MonitoringThresholds(**data)

    def save_thresholds(self, thresholds):
        self.config_path.write_text(
            json.dumps(asdict(thresholds), indent=2)
        )

    def record_prediction(self, result):
        entry = {
            "timestamp": _utc_now(),
            "label": result["label"],
            "label_id": result["label_id"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model_version": result.get("model_version"),
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

        self._trim_log()

    def _trim_log(self):
        thresholds = self.load_thresholds()
        if not self.log_path.exists():
            return

        lines = self.log_path.read_text().splitlines()
        keep = max(thresholds.window_size * 2, thresholds.window_size)
        if len(lines) <= keep:
            return

        trimmed = lines[-keep:]
        self.log_path.write_text("\n".join(trimmed) + "\n")

    def _load_window(self):
        thresholds = self.load_thresholds()
        if not self.log_path.exists():
            return [], thresholds

        lines = self.log_path.read_text().splitlines()
        records = []
        for line in lines[-thresholds.window_size :]:
            if line.strip():
                records.append(json.loads(line))
        return records, thresholds

    def _load_baseline(self):
        metadata_path = self.artifact_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        metadata = json.loads(metadata_path.read_text())
        metrics = metadata.get("metrics", {})
        return {
            "macro_f1": metrics.get("macro_f1"),
            "accuracy": metrics.get("accuracy"),
            "model_version": metadata.get("model_version"),
        }

    def summary(self):
        records, thresholds = self._load_window()
        baseline = self._load_baseline()

        if not records:
            return {
                "prediction_count": 0,
                "window_size": thresholds.window_size,
                "thresholds": asdict(thresholds),
                "baseline": baseline,
                "avg_confidence": None,
                "low_confidence_rate": None,
                "class_counts": {},
                "should_retrain": False,
                "retrain_reasons": ["not enough prediction samples yet"],
            }

        confidences = [row["confidence"] for row in records]
        avg_confidence = sum(confidences) / len(confidences)
        low_confidence_rate = sum(
            1
            for value in confidences
            if value < thresholds.low_confidence_cutoff
        ) / len(confidences)

        class_counts = {}
        for row in records:
            label = row["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

        should_retrain, reasons = self.evaluate(
            thresholds,
            len(records),
            avg_confidence,
            low_confidence_rate,
            baseline,
        )

        return {
            "prediction_count": len(records),
            "window_size": thresholds.window_size,
            "thresholds": asdict(thresholds),
            "baseline": baseline,
            "avg_confidence": round(avg_confidence, 6),
            "low_confidence_rate": round(low_confidence_rate, 6),
            "class_counts": class_counts,
            "should_retrain": should_retrain,
            "retrain_reasons": reasons,
        }

    def evaluate(
        self,
        thresholds,
        sample_count,
        avg_confidence,
        low_confidence_rate,
        baseline,
    ):
        reasons = []

        if sample_count < thresholds.min_samples:
            return False, [
                (
                    f"need at least {thresholds.min_samples} predictions "
                    f"in the monitoring window (have {sample_count})"
                )
            ]

        if avg_confidence < thresholds.min_avg_confidence:
            reasons.append(
                "average confidence "
                f"{avg_confidence:.3f} is below "
                f"{thresholds.min_avg_confidence:.3f}"
            )

        if low_confidence_rate > thresholds.max_low_confidence_rate:
            reasons.append(
                "low-confidence rate "
                f"{low_confidence_rate:.3f} is above "
                f"{thresholds.max_low_confidence_rate:.3f}"
            )

        baseline_f1 = baseline.get("macro_f1")
        f1_drop = float(os.getenv("METRICS_F1_DROP", 0.05))
        if baseline_f1 is not None:
            # Proxy drift check: very low average confidence vs a good baseline F1.
            expected_floor = max(0.35, baseline_f1 - f1_drop)
            if avg_confidence < expected_floor:
                reasons.append(
                    "average confidence looks too low compared with the "
                    f"saved training macro-F1 baseline ({baseline_f1:.3f})"
                )

        if reasons:
            return True, reasons

        return False, ["metrics are inside configured thresholds"]
