import logging
import os
from typing import Any, Callable, List

import dotenv
import fastapi
import mlflow
import mlflow.tracking
import numpy
from pydantic import BaseModel

from ml import Dataset, Learner, models

dotenv.load_dotenv(".env")

mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflowclient = mlflow.tracking.MlflowClient(mlflow.get_tracking_uri(), mlflow.get_registry_uri())

logger = logging.getLogger("uvicorn")
app = fastapi.FastAPI(title="an-experiment")
app.state.cache = {}


def cache(name: str, loader: Callable) -> Any:
    data = app.state.cache.get(name, None)
    if data is None:
        logger.info("Loading %s", name)
        data = loader()
        app.state.cache[name] = data
    return data


@app.get("/api/dataset")
def api_dataset():
    """Dataset stat"""
    dataset = cache("dataset", Dataset.load)
    return {
        "train": {
            "batch_size": dataset.train_dataloader.batch_size,
            "iterations": len(dataset.train_dataloader),
        },
        "val": {
            "batch_size": dataset.val_dataloader.batch_size,
            "iterations": len(dataset.val_dataloader),
        },
    }


class Learning(BaseModel):
    name: str
    hyperparams: dict[str, Any]


@app.post("/api/learn")
def api_learn(learning: Learning, background_tasks: fastapi.BackgroundTasks):
    def run(learning):
        """Background Task"""
        dataset = cache("dataset", Dataset.load)
        mlflow.set_experiment("sample-mnist")

        with mlflow.start_run():
            mlflow.set_tag("mlflow.source.name", os.uname().nodename)
            mlflow.log_params(learning.hyperparams)

            logging.info("Learning")
            model = models.CNN(learning.hyperparams)
            Learner(model).run(dataset, mlflow)

            # Register model
            mlflow.pytorch.log_model(
                model,
                "models",
                registered_model_name=learning.name,
                conda_env=mlflow.pytorch.get_default_conda_env(),
            )
            mv = mlflowclient.search_model_versions(f"name='{learning.name}'")[-1]
            mlflowclient.transition_model_version_stage(
                name=mv.name, version=mv.version, stage="production"
            )

    background_tasks.add_task(run, learning)
    return {"status": "Accepted"}


@app.get("/api/models")
def api_models():
    model_list = mlflowclient.list_registered_models()
    names = [model.name for model in model_list]
    return [
        {
            "name": name,
            "version": mlflowclient.get_registered_model(name).latest_versions[-1].version,
        }
        for name in names
    ]


class Predicting(BaseModel):
    name: str
    version: int
    data: List[List[float]]


@app.post("/api/predict")
async def api_predict(predicting: Predicting):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{predicting.name}/{predicting.version}")
    img = numpy.array(predicting.data, dtype=numpy.float32).flatten()[numpy.newaxis, ...] / 255
    pred = model.predict(img)
    logger.info(pred)
    res = int(numpy.argmax(pred[0]))
    return {"result": res, "prob": pred[0].tolist()}
