set dotenv-load := true

# uvicorn backend
serve:
    uvicorn --host 0.0.0.0 --port ${SERVER_PORT} --reload --reload-dir server --reload-dir ml server.main:app

# streamlit frontend
view:
    streamlit run --server.port ${VIEWER_PORT} --server.headless true viewer/main.py

# mlflow ui
mlflow:
    mlflow ui --backend-store-uri sqlite:///db/backend.db --port $MLFLOW_PORT

# fetch dataset
dataset:
    mkdir -p data/
    hive \
        -e active_dt_begin_hyphen=2021-10-01 \
        -e active_dt_end_hyphen=2021-11-01 \
        -e active_dt_begin=20211001 \
        -e active_dt_end=20211101 \
        -e snapshot_dt=20211101 \
        ./sql/dataset.sql > data/watch.csv
