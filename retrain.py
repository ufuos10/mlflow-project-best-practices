#!/usr/bin/env python
"""Retrain the fake-news text classifier with current library versions.

The original model (saved on Python 3.8 / scikit-learn 0.24.1) cannot be loaded
on Python 3.13. This script trains a similar model and saves it in MLflow
format with the same contract your FastAPI app expects:

    input : DataFrame with one string column named "text"
    output: array of labels ("True" / "False")

Data options (pick one):

    1) Kaggle-style pair of files (columns: title, text, subject, date):
         python retrain.py --true-csv True.csv --fake-csv Fake.csv

    2) One CSV with a "text" column and a "label" column
       (label can be 1/0, true/false or real/fake):
         python retrain.py --data news.csv

    3) A tiny built-in sample, ONLY to check that the pipeline runs:
         python retrain.py --demo

The trained model is written to the folder given by --output (default:
model_new). Point main.py at it with: mlflow.pyfunc.load_model('model_new')
"""
import argparse
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TRUE_LABEL = "True"
FALSE_LABEL = "False"

TRUE_WORDS = {"1", "true", "real", "reliable", "t", "yes"}
FALSE_WORDS = {"0", "false", "fake", "unreliable", "f", "no"}

# Tiny made-up sample. It only proves the code runs. A model trained on it
# is NOT meaningful.
DEMO_REAL = [
    "The central bank held interest rates steady on Tuesday, citing stable inflation.",
    "City officials approved a budget that increases funding for public transport.",
    "The university announced a new scholarship program for engineering students.",
    "Researchers published a peer reviewed study on soil moisture and crop yields.",
    "The national football team will play two friendly matches next month.",
    "Parliament passed a bill updating the rules for small business registration.",
    "A new bridge opened to traffic after two years of construction.",
    "The health ministry reported a decline in seasonal flu cases this week.",
]
DEMO_FAKE = [
    "SHOCKING: scientists admit the moon landing was staged, share before they delete this!",
    "Doctors hate this one weird trick, the government is hiding the miracle cure!",
    "BREAKING: secret group controls the weather and they just confessed on video!",
    "You will not believe what they put in tap water, wake up people!!!",
    "Celebrity found alive after fake death, the media does not want you to know!",
    "Leaked memo PROVES everything you were told is a lie, share now!",
    "Aliens signed a secret treaty with world leaders, insider reveals all!",
    "Eat this fruit every morning and never get sick again, big pharma furious!",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--true-csv", help="CSV of real news (Kaggle True.csv style)")
    parser.add_argument("--fake-csv", help="CSV of fake news (Kaggle Fake.csv style)")
    parser.add_argument("--data", help="single CSV with 'text' and 'label' columns")
    parser.add_argument("--demo", action="store_true",
                        help="use a tiny built-in sample (pipeline check only)")
    parser.add_argument("--output", default="model_new", help="folder for the saved model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(value):
    """Map many label spellings to "True" / "False"."""
    text = str(value).strip().lower()
    if text in TRUE_WORDS:
        return TRUE_LABEL
    if text in FALSE_WORDS:
        return FALSE_LABEL
    raise ValueError(f"Unrecognised label value: {value!r}")


def load_data(args):
    """Return a DataFrame with columns: text, label."""
    if args.demo:
        print("WARNING: using the tiny demo sample. The resulting model is not meaningful.")
        texts = DEMO_REAL + DEMO_FAKE
        labels = [TRUE_LABEL] * len(DEMO_REAL) + [FALSE_LABEL] * len(DEMO_FAKE)
        return pd.DataFrame({"text": texts, "label": labels})

    if args.true_csv and args.fake_csv:
        real = pd.read_csv(args.true_csv)
        fake = pd.read_csv(args.fake_csv)
        for name, frame in (("true-csv", real), ("fake-csv", fake)):
            if "text" not in frame.columns:
                sys.exit(f"--{name} needs a 'text' column. Found: {list(frame.columns)}")
        real = real[["text"]].assign(label=TRUE_LABEL)
        fake = fake[["text"]].assign(label=FALSE_LABEL)
        return pd.concat([real, fake], ignore_index=True)

    if args.data:
        frame = pd.read_csv(args.data)
        missing = {"text", "label"} - set(frame.columns)
        if missing:
            sys.exit(f"--data is missing column(s): {sorted(missing)}. "
                     f"Found: {list(frame.columns)}")
        frame = frame[["text", "label"]].copy()
        frame["label"] = frame["label"].map(normalize_label)
        return frame

    sys.exit("No data given. Use --true-csv and --fake-csv, or --data, or --demo. "
             "Run with --help for details.")


def build_pipeline():
    """TF-IDF on the 'text' column followed by logistic regression."""
    features = ColumnTransformer(
        [("tfidf",
          TfidfVectorizer(stop_words="english", max_features=20000),
          "text")]
    )
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    args = parse_args()

    df = load_data(args)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""].drop_duplicates(subset="text")
    print(f"Loaded {len(df)} rows. Label counts:\n{df['label'].value_counts()}")

    x_train, x_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"],
        test_size=args.test_size, random_state=args.seed, stratify=df["label"],
    )

    pipeline = build_pipeline()

    # Newer MLflow refuses the old ./mlruns folder backend, so log to SQLite.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label=TRUE_LABEL)
        print(f"Accuracy: {accuracy:.3f}   F1 (class '{TRUE_LABEL}'): {f1:.3f}")

        mlflow.log_params({"model": "tfidf+logreg", "rows": len(df),
                           "test_size": args.test_size, "seed": args.seed})
        mlflow.log_metrics({"accuracy": accuracy, "f1": f1})

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)

    signature = infer_signature(x_test, preds)
    mlflow.sklearn.save_model(
        pipeline,
        path=str(output),
        signature=signature,
        input_example=x_test.head(2),
        serialization_format="cloudpickle",
    )
    print(f"Saved model to: {output}")

    # Sanity check: reload through the same API that main.py uses.
    loaded = mlflow.pyfunc.load_model(str(output))
    sample = pd.DataFrame({"text": {0: "The city council approved the new budget on Monday."}})
    print("Reload check, sample prediction:", loaded.predict(sample).tolist())


if __name__ == "__main__":
    main()