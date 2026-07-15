"""ML scoring layer: Isolation Forest + SHAP for anomaly detection."""
import logging
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import IsolationForest

from detector.features import FeatureVector, FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class AnomalyScorer:
    """Wraps Isolation Forest with SHAP-based explainability."""

    def __init__(self, contamination: float = 0.05):
        self.model: IsolationForest | None = None
        self.contamination = contamination
        self._explainer = None
        self._is_loaded = False

    def train(self, features: np.ndarray) -> dict:
        """Train Isolation Forest on a feature matrix.

        Args:
            features: numpy array of shape (n_samples, n_features)
        Returns:
            Dict with training metrics
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        self.model.fit(features)

        scores = self._raw_scores(features)
        metrics = {
            "n_samples": int(features.shape[0]),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "p95_score": float(np.percentile(scores, 95)),
            "p99_score": float(np.percentile(scores, 99)),
            "contamination": self.contamination,
        }
        logger.info(f"Model trained: {metrics}")
        return metrics

    def predict(self, feature_vector: FeatureVector) -> tuple[float, dict]:
        """Score a single feature vector.

        Returns:
            (anomaly_score 0-1, shap_values dict)
        """
        if self.model is None:
            return 0.0, {}

        arr = np.array([feature_vector.to_array()])
        raw = self._raw_scores(arr)[0]

        # Normalize: Isolation Forest score is typically in [-1, 0] for anomalies
        # and [0, 1] for normal. We invert and normalize to 0-1 where higher = more anomalous.
        score = max(0.0, min(1.0, -raw))

        shap_values = self._compute_shap(feature_vector)
        return float(score), shap_values

    def _raw_scores(self, features: np.ndarray) -> np.ndarray:
        """Get raw anomaly scores (decision_function)."""
        if self.model is None:
            return np.zeros(features.shape[0])
        return self.model.decision_function(features)

    def _compute_shap(self, feature_vector: FeatureVector) -> dict:
        """Compute SHAP values for feature-level explanations."""
        if self.model is None:
            return {}

        try:
            import shap

            if self._explainer is None:
                # Use TreeExplainer for Isolation Forest
                self._explainer = shap.TreeExplainer(self.model)

            arr = np.array([feature_vector.to_array()])
            shap_vals = self._explainer.shap_values(arr)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            shap_vals = np.abs(shap_vals[0])
            total = shap_vals.sum()
            if total > 0:
                normalized = shap_vals / total
            else:
                normalized = shap_vals

            return {
                name: float(val)
                for name, val in zip(FEATURE_NAMES, normalized)
            }
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {}

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "contamination": self.contamination,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.contamination = data.get("contamination", self.contamination)
            self._explainer = None  # Reset explainer for new model
            self._is_loaded = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self.model is not None
