"""Synthetic, in-memory model: no external data or model downloads."""

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Small synthetic training set, deliberately not presented as production quality.
TRAIN = {
    "billing": [
        "invoice has the wrong amount",
        "payment failed on my subscription",
        "please refund my duplicate charge",
        "billing address and receipt",
        "price and tax on invoice",
        "charged twice by card",
        "payment method expired",
        "monthly bill cost is incorrect",
    ],
    "account": [
        "cannot log in to account",
        "password reset link expired",
        "enable two factor authentication",
        "locked out after login attempts",
        "change my account email",
        "account access denied",
        "forgot password and username",
        "authentication code not working",
    ],
    "technical": [
        "application crashes when uploading file",
        "api returns server error",
        "dashboard page is broken",
        "export download does not work",
        "service timeout on requests",
        "bug in software after update",
        "integration returns error",
        "file upload fails repeatedly",
    ],
}
CATEGORIES = tuple(TRAIN)


@lru_cache(maxsize=1)
def model():
    x, y = [], []
    for label, samples in TRAIN.items():
        x.extend(samples)
        y.extend([label] * len(samples))
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    return pipeline.fit(x, y)


def predict(text: str) -> dict:
    pipeline = model()
    probabilities = pipeline.predict_proba([text])[0]
    index = int(probabilities.argmax())
    label = str(pipeline.classes_[index])
    confidence = float(probabilities[index])
    return {
        "category": label,
        "confidence": round(confidence, 4),
        "requires_review": confidence < 0.55,
    }
