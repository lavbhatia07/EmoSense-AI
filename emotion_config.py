from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

LEGACY_MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
LEGACY_VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

LABEL_TO_ID = {
    "sadness": 0,
    "anger": 1,
    "love": 2,
    "surprise": 3,
    "fear": 4,
    "joy": 5,
}

EMOTION_META = {
    "sadness": {
        "emoji": "😢",
        "color": "#4d8dff",
        "description": "The text carries a reflective, heavy, or emotionally low tone.",
    },
    "anger": {
        "emoji": "😡",
        "color": "#ff5b5b",
        "description": "The language suggests frustration, conflict, irritation, or intense negativity.",
    },
    "love": {
        "emoji": "💖",
        "color": "#ff69b4",
        "description": "The message feels affectionate, caring, warm, or emotionally attached.",
    },
    "surprise": {
        "emoji": "😲",
        "color": "#ffd84d",
        "description": "The wording shows amazement, shock, sudden curiosity, or unexpected excitement.",
    },
    "fear": {
        "emoji": "😨",
        "color": "#a56bff",
        "description": "The text reflects worry, uncertainty, anxiety, stress, or emotional tension.",
    },
    "joy": {
        "emoji": "😊",
        "color": "#ff9f43",
        "description": "The overall feeling is cheerful, positive, relieved, optimistic, or celebratory.",
    },
}

DEFAULT_EMOTION_DETAILS = {
    "emoji": "🤖",
    "color": "#00d4ff",
    "description": "The model returned an emotion outside the standard label set.",
}
