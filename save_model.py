from challenge.model import DelayModel
import pandas as pd

model = DelayModel()
data = pd.read_csv(filepath_or_buffer="./data/data.csv")

features, target = model.preprocess(data=data, target_column="delay")

model.fit(features=features, target=target)

model.save_model("./data/model.pkl")
