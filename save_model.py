"""Train and save the flight delay prediction model."""

import pandas as pd

from challenge.model import DelayModel


def main() -> None:
    model = DelayModel()
    data = pd.read_csv("./data/data.csv")
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)
    model.save_model("./data/model.pkl")


if __name__ == "__main__":
    main()
