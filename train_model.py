from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from emotion_config import DATA_DIR, LABEL_TO_ID, MODELS_DIR
from text_processing import preprocess_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("trainer")

DATASET_PATH = Path(os.getenv("DATASET_FILE", DATA_DIR / "emotion_dataset.txt"))


def load_dataset(file_path: Path) -> tuple[list[str], list[int]]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Add a dataset file before training."
        )

    texts: list[str] = []
    labels: list[int] = []

    with file_path.open("r", encoding="utf-8") as file:
        sample = file.read(2048)
        file.seek(0)

        if ";" in sample and "\n" in sample and "text" not in sample.lower():
            for line in file:
                row = line.strip()
                if not row or ";" not in row:
                    continue
                text, emotion = row.rsplit(";", 1)
                emotion = emotion.strip().lower()
                if emotion not in LABEL_TO_ID:
                    continue
                cleaned_text = preprocess_text(text)
                if cleaned_text:
                    texts.append(cleaned_text)
                    labels.append(LABEL_TO_ID[emotion])
        else:
            reader = csv.DictReader(file)
            lowered_fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
            text_column = next(
                (lowered_fieldnames[name] for name in ["text", "sentence", "content", "comment", "review"] if name in lowered_fieldnames),
                None,
            )
            label_column = next(
                (lowered_fieldnames[name] for name in ["emotion", "label", "sentiment", "target"] if name in lowered_fieldnames),
                None,
            )
            if not text_column or not label_column:
                raise ValueError("Dataset must contain text and emotion columns.")

            for row in reader:
                emotion = str(row[label_column]).strip().lower()
                if emotion not in LABEL_TO_ID:
                    continue
                cleaned_text = preprocess_text(str(row[text_column]))
                if cleaned_text:
                    texts.append(cleaned_text)
                    labels.append(LABEL_TO_ID[emotion])

    if not texts:
        raise ValueError("No usable rows were found in the dataset.")

    return texts, labels


def build_candidates() -> list[tuple[str, object, object]]:
    vectorizers = {
        "tfidf_word_ngrams": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
        "count_plus_tfidf": Pipeline(
            [
                ("count", CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
                ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ]
        ),
    }

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            C=4.0,
            class_weight="balanced",
            solver="liblinear",
        ),
        "naive_bayes": MultinomialNB(alpha=0.4),
        "linear_svm": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.5, class_weight="balanced"),
            cv=3,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        ),
    }

    candidates: list[tuple[str, object, object]] = []
    for vectorizer_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            candidates.append((f"{vectorizer_name}__{model_name}", vectorizer, model))
    return candidates


def evaluate_candidates(x_train: list[str], x_test: list[str], y_train: list[int], y_test: list[int]) -> dict:
    best_result: dict | None = None
    leaderboard: list[dict] = []

    for candidate_name, vectorizer, model in build_candidates():
        LOGGER.info("Training candidate: %s", candidate_name)
        x_train_vectorized = vectorizer.fit_transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        model.fit(x_train_vectorized, y_train)
        predictions = model.predict(x_test_vectorized)
        accuracy = accuracy_score(y_test, predictions)

        result = {
            "candidate_name": candidate_name,
            "vectorizer": vectorizer,
            "model": model,
            "accuracy": accuracy,
            "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        }
        leaderboard.append({"candidate_name": candidate_name, "accuracy": round(accuracy, 4)})

        LOGGER.info("Candidate %s accuracy: %.4f", candidate_name, accuracy)
        if best_result is None or accuracy > best_result["accuracy"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("No candidate models were trained.")

    best_result["leaderboard"] = sorted(leaderboard, key=lambda item: item["accuracy"], reverse=True)
    return best_result


def save_artifacts(best_result: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dump(best_result["model"], MODELS_DIR / "best_model.joblib")
    dump(best_result["vectorizer"], MODELS_DIR / "vectorizer.joblib")

    with (MODELS_DIR / "label_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(LABEL_TO_ID, file, indent=2)

    with (MODELS_DIR / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_candidate": best_result["candidate_name"],
                "best_accuracy": round(best_result["accuracy"], 4),
                "leaderboard": best_result["leaderboard"],
                "classification_report": best_result["report"],
            },
            file,
            indent=2,
        )


def main() -> None:
    texts, labels = load_dataset(DATASET_PATH)
    LOGGER.info("Loaded %s training rows from %s", len(texts), DATASET_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    best_result = evaluate_candidates(x_train, x_test, y_train, y_test)
    save_artifacts(best_result)

    LOGGER.info("Best candidate: %s", best_result["candidate_name"])
    LOGGER.info("Best accuracy: %.4f", best_result["accuracy"])
    LOGGER.info("Artifacts saved to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
