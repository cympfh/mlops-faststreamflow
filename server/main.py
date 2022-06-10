import logging
import os
from typing import Any, Callable, List, Optional, TypeVar

import dotenv
import fastapi
import mlflow
import mlflow.tracking
import numpy
from pydantic import BaseModel

from ml import Dataset, Learner, models
from server.util import mutex

dotenv.load_dotenv(".env")

mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflowclient = mlflow.tracking.MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri()
)

logger = logging.getLogger("uvicorn.app")
app = fastapi.FastAPI(title=os.environ.get("EXPERIMENT_NAME", "app"))
app.state.cache = {}


T_Cache = TypeVar("T_Cache")


def cache(name: str, loader: Callable[[], T_Cache]) -> T_Cache:
    data: Optional[T_Cache] = app.state.cache.get(name, None)
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


@mutex
def log_model(model, name, run):
    logger.info("Logging Model name=%s, run_id=%s", name, run.info.run_id)
    mlflow.tracking.fluent._active_run_stack = [run]
    mlflow.pytorch.log_model(
        model,
        f"runs:/{run.info.run_id}/model",
        registered_model_name=name,
    )
    mv = mlflowclient.search_model_versions(f"name='{name}'")[-1]
    mlflowclient.transition_model_version_stage(
        name=mv.name, version=mv.version, stage="production"
    )


@app.post("/api/learn")
def api_learn(learning: Learning, background_tasks: fastapi.BackgroundTasks):
    def task(learning):
        """Background Task"""

        # Start up
        experiment = mlflow.set_experiment(os.environ.get("EXPERIMENT_NAME"))
        run = mlflowclient.create_run(experiment.experiment_id)
        logger.info(
            "Experiment: experiment_id=%s, run_id=%s",
            experiment.experiment_id,
            run.info.run_id,
        )

        mlflowclient.set_tag(run.info.run_id, "mlflow.source.name", os.uname().nodename)
        for key, val in learning.hyperparams.items():
            mlflowclient.log_param(run.info.run_id, key, val)

        # Learning
        dataset = cache("dataset", Dataset.load)
        model = models.CNN(learning.hyperparams)
        Learner(model).run(dataset, mlflowclient, run.info.run_id)

        # Close up
        log_model(model, learning.name, run)
        mlflowclient.set_terminated(run.info.run_id)

    background_tasks.add_task(task, learning)
    return {"status": "Accepted"}


@app.get("/api/models")
def api_models():
    model_list = mlflowclient.list_registered_models()
    names = [model.name for model in model_list]
    return [
        {
            "name": name,
            "version": mlflowclient.get_registered_model(name)
            .latest_versions[-1]
            .version,
        }
        for name in names
    ]


class Predicting(BaseModel):
    name: str
    version: int
    data: List[List[float]]


@app.post("/api/predict")
async def api_predict(predicting: Predicting):
    logger.info("Predicting with %s/%s", predicting.name, predicting.version)
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{predicting.name}/{predicting.version}"
    )
    logger.info("%s", model)
    img = (
        numpy.array(predicting.data, dtype=numpy.float32).flatten()[numpy.newaxis, ...]
        / 255
    )
    img = -img + 1.0
    pred = model.predict(img)
    res = int(numpy.argmax(pred[0]))
    logger.info("Pred is %s from %s", res, pred)
    return {"result": res, "prob": pred[0].tolist()}
