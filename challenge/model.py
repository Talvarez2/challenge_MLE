"""Flight delay prediction model."""

import pickle
from pathlib import Path
from typing import Union

import pandas as pd
import xgboost as xgb

from challenge.preprocessing import preprocess


class DelayModel:
    """XGBoost classifier for predicting flight delays at SCL airport."""

    def __init__(self) -> None:
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._fitted = False

    def preprocess(
        self, data: pd.DataFrame, target_column: str | None = None
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Prepare raw data for training or prediction.

        Args:
            data: Raw flight data.
            target_column: If set, the target column is returned alongside features.

        Returns:
            Features DataFrame, or (features, target) tuple.
        """
        return preprocess(data=data, target_column=target_column)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Fit the model with preprocessed data.

        Args:
            features: Preprocessed feature matrix.
            target: Target labels.
        """
        y_train = target.squeeze()
        n_y0 = int((y_train == 0).sum())
        n_y1 = int((y_train == 1).sum())
        self._model.scale_pos_weight = n_y0 / n_y1

        self._model.fit(features, y_train)
        self._fitted = True

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Predict delays for new flights.

        Args:
            features: Preprocessed feature matrix.

        Returns:
            List of predicted labels (0 = on time, 1 = delayed).
        """
        if not self._fitted:
            return [0] * len(features)

        return self._model.predict(features).tolist()

    def save_model(self, path: str) -> None:
        """Serialize the trained model to disk.

        Args:
            path: Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
