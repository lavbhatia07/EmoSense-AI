from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from joblib import load as joblib_load
from scipy.special import softmax
from sklearn.feature_extraction.text import HashingVectorizer

from emotion_config import (
    DEFAULT_EMOTION_DETAILS,
    EMOTION_META,
    LABEL_TO_ID,
    LEGACY_MODEL_PATH,
    LEGACY_VECTORIZER_PATH,
    MODELS_DIR,
)
from text_processing import preprocess_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
LABEL_MAP_PATH = MODELS_DIR / "label_mapping.json"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_pickle(file_path: Path) -> Any:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        return joblib_load(file_path)
    except Exception:
        with file_path.open("rb") as file:
            return pickle.load(file)


class EmotionModelService:
    """Load model artifacts and run consistent prediction logic."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("emotion_service")
        self.model: Any | None = None
        self.vectorizer: Any | None = None
        self.id_to_label = {value: key for key, value in LABEL_TO_ID.items()}
        self.model_source = "unavailable"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if MODEL_PATH.exists() and VECTORIZER_PATH.exists() and LABEL_MAP_PATH.exists():
            self.model = load_pickle(MODEL_PATH)
            self.vectorizer = load_pickle(VECTORIZER_PATH)
            with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
                label_mapping = json.load(file)
            self.id_to_label = {int(value): key for key, value in label_mapping.items()}
            self.model_source = "trained-artifacts"
            self.logger.info("Loaded trained artifacts from %s", MODELS_DIR)
            return

        if LEGACY_MODEL_PATH.exists():
            self.model = load_pickle(LEGACY_MODEL_PATH)
            self.vectorizer = load_pickle(LEGACY_VECTORIZER_PATH) if LEGACY_VECTORIZER_PATH.exists() else None
            self.model_source = "legacy-fallback"
            self.logger.warning(
                "Using legacy fallback model. Retrain with train_model.py for best accuracy and consistent preprocessing."
            )
            return

        raise FileNotFoundError(
            "No trained artifacts were found. Run train_model.py after placing a dataset in the data folder."
        )

    def _vectorize(self, cleaned_text: str) -> Any:
        if self.vectorizer is not None:
            vectorized = self.vectorizer.transform([cleaned_text])
        else:
            feature_count = getattr(self.model, "n_features_in_", 4096)
            fallback_vectorizer = HashingVectorizer(
                n_features=feature_count if isinstance(feature_count, int) and feature_count > 0 else 4096,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
            )
            vectorized = fallback_vectorizer.transform([cleaned_text])

        self.logger.info("Vectorized shape: %s", getattr(vectorized, "shape", "unknown"))
        return vectorized

    def _decode_prediction(self, prediction: Any) -> str:
        try:
            numeric_prediction = int(prediction)
            return self.id_to_label.get(numeric_prediction, str(numeric_prediction))
        except (TypeError, ValueError):
            return str(prediction).strip().lower()

    def _build_scores(self, vectorized: Any) -> tuple[dict[str, float] | None, float | None, Any]:
        raw_prediction = self.model.predict(vectorized)[0]
        self.logger.info("Raw prediction: %s", raw_prediction)

        scores = None
        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vectorized)[0]
            labels = [self._decode_prediction(label) for label in getattr(self.model, "classes_", [])]
            scores = {
                label: round(float(probability) * 100, 2)
                for label, probability in zip(labels, probabilities, strict=False)
            }
            confidence = max(scores.values()) if scores else None
        elif hasattr(self.model, "decision_function"):
            decision_scores = self.model.decision_function(vectorized)
            probabilities = softmax(np.ravel(decision_scores))
            labels = [self._decode_prediction(label) for label in getattr(self.model, "classes_", range(len(probabilities)))]
            scores = {
                label: round(float(probability) * 100, 2)
                for label, probability in zip(labels, probabilities, strict=False)
            }
            confidence = max(scores.values()) if scores else None

        self.logger.info("Confidence score: %s", confidence if confidence is not None else "not-available")
        return scores, confidence, raw_prediction

    def predict(self, raw_text: str) -> dict[str, Any]:
        cleaned_text = preprocess_text(raw_text)
        self.logger.info("Cleaned text: %s", cleaned_text)

        if not cleaned_text:
            raise ValueError("Please enter some text before analyzing.")

        vectorized = self._vectorize(cleaned_text)
        scores, confidence, raw_prediction = self._build_scores(vectorized)
        emotion = self._decode_prediction(raw_prediction)
        emotion_details = EMOTION_META.get(emotion, DEFAULT_EMOTION_DETAILS)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
            "emoji": emotion_details["emoji"],
            "color": emotion_details["color"],
            "description": emotion_details["description"],
            "cleaned_text": cleaned_text,
            "model_source": self.model_source,
        }


configure_logging()
app = Flask(__name__)
emotion_service = EmotionModelService()


@app.get("/")
def index():
    return render_template(
        "index.html",
        emotions=EMOTION_META,
        model_source=emotion_service.model_source,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_source": emotion_service.model_source})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        result = emotion_service.predict(text)
        return jsonify({"success": True, "result": result})
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"success": False, "error": str(error)}), 500
    except Exception as error:
        logging.getLogger("emotion_service").exception("Prediction failed")
        message = "Prediction failed. Retrain the model or verify the saved artifacts."
        if os.getenv("FLASK_ENV") == "development":
            message = f"{message} Details: {error}"
        return jsonify({"success": False, "error": message}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_mode)
