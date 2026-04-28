"""Train and save the flight delay prediction model."""

import pandas as pd

from challenge.config import MODEL_PATH
from challenge.model import DelayModel


def main() -> None:
    """Train the delay model on the full dataset and persist it."""
    data = pd.read_csv("./data/data.csv")
    model = DelayModel()
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)
    model.save_model(MODEL_PATH)


if __name__ == "__main__":
    main()
