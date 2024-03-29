import pandas as pd
import fastapi
import pickle
from mangum import Mangum
from contextlib import asynccontextmanager
from challenge.preprossessing_functions import preprocess

file_name = "./data/model.pkl"
model = pickle.load(open(file_name, "rb"))


@asynccontextmanager
async def lifespan(application: fastapi.FastAPI):
    # data = await pd.read_csv(filepath_or_buffer="./data/data.csv")
    # features, target = await model.preprocess(data=data, target_column="delay")
    # await model.fit(features, target)
    yield
    print("Cleaning up")


app = fastapi.FastAPI()
handler = Mangum(app)


def check_valid_flight(flight):
    if "OPERA" not in flight:
        raise Exception("OPERA column is missing")
    elif "TIPOVUELO" not in flight:
        raise Exception("TIPOVUELO column is missing")
    elif flight["TIPOVUELO"] not in ["N", "I"]:
        raise Exception("TIPOVUELO value is not valid")
    elif "MES" not in flight:
        raise Exception("MES column is missing")
    elif flight["MES"] not in range(1, 13):
        raise Exception("MES value is not valid")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    try:
        flights = data["flights"]
        print(flights)
        for flight in flights:
            check_valid_flight(flight)
        features = preprocess(pd.DataFrame(flights, index=[0]))
        print(features)
        prediction = model.predict(features)

        return {"predict": prediction.tolist()}

    except Exception as e:
        print(e)
        raise fastapi.HTTPException(status_code=400, detail=str(e))
