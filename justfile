set dotenv-load := true

# uvicorn backend
serve:
    uvicorn \
        --host 0.0.0.0 \
        --port ${SERVER_PORT} \
        --reload \
        --reload-dir server \
        --reload-dir ml \
        --log-config logconf.yml \
        server.main:app

# streamlit frontend
view:
    streamlit run --server.port ${VIEWER_PORT} --server.headless true viewer/main.py

# mlflow ui
mlflow:
    mlflow ui --backend-store-uri sqlite:///db/backend.db --port $MLFLOW_PORT
