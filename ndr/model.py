"""
The classifier and the end-to-end resolver.

Model choice
------------
Word 1-2 grams + character 2-5 grams (char_wb), TF-IDF weighted, into a
multinomial logistic regression.

Character n-grams are doing the heavy lifting, not the word n-grams. A reply
containing "dlvry", "delivry" or "dilivery" produces almost no useful word
features, but shares the char-grams 'liv', 'lvr', 'vry' with every other
spelling of the word. This is what makes the model robust to spellings that
were never seen in training — which, for user-generated Hinglish, is most of
them. It is also why a transformer is not obviously the right first move here:
pretrained subword vocabularies are built for well-formed text, most Indic
models are not trained on romanized Hindi, and this pipeline classifies a reply
in well under a millisecond on CPU with no GPU bill at NDR volume.

Logistic regression (rather than SVM) because the decision layer needs
calibrated-ish probabilities to abstain on, and because per-class coefficients
are directly inspectable when ops asks "why did it say that".
"""

from __future__ import annotations

from dataclasses import asdict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .normalize import preprocess
from .rules import apply_rules
from .taxonomy import Decision, decide

# Confidence assigned when a rule is trusted over the model.
RULE_OVERRIDE_CONFIDENCE = 0.70
# Cap applied to model confidence when a strong rule disagrees with it.
DISAGREEMENT_CAP = 0.55


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                sublinear_tf=True, token_pattern=r"(?u)<?\b\w+\b>?",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                sublinear_tf=True,
            )),
        ])),
        ("clf", LogisticRegression(
            C=8.0, max_iter=3000, class_weight="balanced",
        )),
    ])


class NDRResolver:
    """Normalize -> classify -> fuse with rules -> map to an operational action."""

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or build_pipeline()
        self.classes_: np.ndarray | None = None

    # ----------------------------------------------------------------- train
    def fit(self, texts, labels) -> "NDRResolver":
        normalized = [preprocess(t)[0] for t in texts]
        self.pipeline.fit(normalized, labels)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba([preprocess(t)[0] for t in texts])

    def predict(self, texts) -> list[str]:
        return list(self.pipeline.predict([preprocess(t)[0] for t in texts]))

    # --------------------------------------------------------------- resolve
    def resolve(self, text: str) -> Decision:
        normalized, entities = preprocess(text)

        if not normalized:
            return decide("UNCLEAR", 1.0, entities)

        proba = self.pipeline.predict_proba([normalized])[0]
        order = np.argsort(proba)[::-1]
        top_intent = str(self.classes_[order[0]])
        top_conf = float(proba[order[0]])
        alternatives = [
            {"intent": str(self.classes_[i]), "confidence": round(float(proba[i]), 4)}
            for i in order[1:4]
        ]

        rule_intent, rule_strength, matched = apply_rules(normalized)
        rule_note = ""

        if rule_intent:
            if rule_intent == top_intent:
                top_conf = min(1.0, top_conf + 0.10)
                rule_note = f" Rule agreement ({rule_strength})."
            elif rule_strength == "strong":
                if top_conf < RULE_OVERRIDE_CONFIDENCE:
                    rule_note = (f" Strong rule '{matched[0]}' overrode model "
                                 f"('{top_intent}' @ {top_conf:.2f}).")
                    top_intent, top_conf = rule_intent, RULE_OVERRIDE_CONFIDENCE
                else:
                    # Both are confident and they disagree -> genuinely ambiguous.
                    top_conf = min(top_conf, DISAGREEMENT_CAP)
                    rule_note = (f" Model/rule disagreement (rule says "
                                 f"'{rule_intent}'); confidence capped for review.")

        decision = decide(top_intent, top_conf, entities, alternatives)
        decision.reason += rule_note
        return decision

    def resolve_batch(self, texts) -> list[dict]:
        out = []
        for t in texts:
            d = self.resolve(t)
            row = asdict(d)
            row["text"] = t
            out.append(row)
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str) -> "NDRResolver":
        blob = joblib.load(path)
        obj = cls(blob["pipeline"])
        obj.classes_ = blob["classes"]
        return obj

    # -------------------------------------------------------- explainability
    def top_features(self, intent: str, n: int = 12) -> list[tuple[str, float]]:
        """Highest-weighted word features for an intent — for ops trust/debugging."""
        clf = self.pipeline.named_steps["clf"]
        union = self.pipeline.named_steps["features"]
        word_vec = union.transformer_list[0][1]
        names = word_vec.get_feature_names_out()
        idx = list(clf.classes_).index(intent)
        coefs = clf.coef_[idx][:len(names)]
        top = np.argsort(coefs)[::-1][:n]
        return [(names[i], round(float(coefs[i]), 3)) for i in top]
