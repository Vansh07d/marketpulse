from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "finbert_model"

MAX_LENGTH = 128


# ============================================================
# LABEL CONFIGURATION
# ============================================================

LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# FINBERT SENTIMENT SERVICE
# ============================================================

class FinBERTSentiment:
    """
    Handles inference using the trained MarketPulse FinBERT model.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        max_length: int = MAX_LENGTH,
    ):

        self.model_path = Path(model_path)
        self.max_length = max_length

        # ----------------------------------------------------
        # Verify model exists
        # ----------------------------------------------------

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"\nFinBERT model not found at:\n"
                f"{self.model_path}\n\n"
                f"Expected structure:\n"
                f"models/\n"
                f"└── finbert_model/\n"
                f"    ├── config.json\n"
                f"    ├── model.safetensors\n"
                f"    └── tokenizer files\n"
            )

        # ----------------------------------------------------
        # Select device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"\nUsing device: {self.device}"
        )

        if torch.cuda.is_available():

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        print(
            "\nLoading FinBERT tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                str(self.model_path)
            )
        )

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        print(
            "Loading trained FinBERT model..."
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                str(self.model_path)
            )
        )

        self.model.to(self.device)

        # ----------------------------------------------------
        # Inference mode
        # ----------------------------------------------------

        self.model.eval()

        print(
            "FinBERT loaded successfully."
        )

    # ========================================================
    # SINGLE TEXT PREDICTION
    # ========================================================

    def predict(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Predict sentiment for a single financial text.

        Returns:

        {
            "text": "...",
            "label": "positive",
            "negative": 0.01,
            "neutral": 0.05,
            "positive": 0.94,
            "score": 0.93
        }
        """

        if not isinstance(text, str):

            raise TypeError(
                "text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "text cannot be empty."
            )

        results = self.predict_batch(
            [text]
        )

        return results[0]

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        texts: list[str],
    ) -> list[dict[str, Any]]:
        """
        Run FinBERT on multiple texts simultaneously.

        Batch inference is important for MarketPulse because
        several news headlines may arrive at once.
        """

        if not texts:

            return []

        cleaned_texts = []

        for text in texts:

            if not isinstance(text, str):

                raise TypeError(
                    "Every item in texts must be a string."
                )

            text = text.strip()

            if not text:

                raise ValueError(
                    "texts cannot contain empty strings."
                )

            cleaned_texts.append(text)

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        encoded = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move tensors to CPU/GPU
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = self.model(
                **encoded
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        probabilities = (
            probabilities
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Convert predictions
        # ----------------------------------------------------

        results = []

        for text, probs in zip(
            cleaned_texts,
            probabilities,
        ):

            negative_probability = float(
                probs[0]
            )

            neutral_probability = float(
                probs[1]
            )

            positive_probability = float(
                probs[2]
            )

            predicted_class = int(
                torch.argmax(probs).item()
            )

            label = LABELS[
                predicted_class
            ]
            score = (
                positive_probability
                - negative_probability
            )

            results.append(
                {
                    "text": text,
                    "label": label,
                    "negative": negative_probability,
                    "neutral": neutral_probability,
                    "positive": positive_probability,
                    "score": score,
                }
            )

        return results


# ============================================================
# SHARED FINBERT INSTANCE
# ============================================================

_finbert_instance: FinBERTSentiment | None = None


def get_finbert() -> FinBERTSentiment:
    """
    Return a shared FinBERT instance.

    The model is loaded only once instead of being loaded every
    time a prediction is requested.
    """

    global _finbert_instance

    if _finbert_instance is None:

        _finbert_instance = FinBERTSentiment()

    return _finbert_instance


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def predict_sentiment(
    text: str,
) -> dict[str, Any]:
    """
    Predict sentiment for one piece of text.
    """

    return get_finbert().predict(text)


def predict_sentiments(
    texts: list[str],
) -> list[dict[str, Any]]:
    """
    Predict sentiment for multiple texts.
    """

    return get_finbert().predict_batch(
        texts
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "MarketPulse FinBERT Inference Test"
    )

    print(
        "========================================\n"
    )

    print(
        f"Project directory:\n{BASE_DIR}"
    )

    print(
        f"\nModel directory:\n{MODEL_PATH}"
    )

    test_headlines = [

        "Infosys reports strong quarterly revenue growth",

        "Tata Motors shares fall after weak earnings",

        "Markets remain stable as investors await new data",

    ]

    results = predict_sentiments(
        test_headlines
    )

    print(
        "\n========================================"
    )

    print(
        "Predictions"
    )

    print(
        "========================================"
    )

    for result in results:

        print(
            f"\nHeadline:"
        )

        print(
            f"  {result['text']}"
        )

        print(
            f"\nSentiment:"
        )

        print(
            f"  {result['label'].upper()}"
        )

        print(
            f"\nProbabilities:"
        )

        print(
            f"  Negative: "
            f"{result['negative']:.4f}"
        )

        print(
            f"  Neutral:  "
            f"{result['neutral']:.4f}"
        )

        print(
            f"  Positive: "
            f"{result['positive']:.4f}"
        )

        print(
            f"\nSentiment score:"
        )

        print(
            f"  {result['score']:+.4f}"
        )

    print(
        "\n========================================"
    )

    print(
        "Inference test completed."
    )

    print(
        "========================================\n"
    )