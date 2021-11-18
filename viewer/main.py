import enum
import json

import cv2
import dotenv
import graphviz
import requests
import streamlit
from streamlit_drawable_canvas import st_canvas


class API:
    @classmethod
    def url(cls, uri: str) -> str:
        port = dotenv.get_key(".env", "SERVER_PORT")
        delimiter = "" if uri.startswith("/") else "/"
        return f"http://localhost:{port}{delimiter}{uri}"

    @classmethod
    def get(cls, uri: str, params: dict = {}):
        res = requests.get(cls.url(uri), params)
        if not res.ok:
            res.raise_for_status()
        return res

    @classmethod
    def post(cls, uri: str, params: dict = {}):
        res = requests.post(cls.url(uri), data=json.dumps(params))
        if not res.ok:
            res.raise_for_status()
        return res


class Task(enum.Enum):
    LEARN = "Learn"
    PREDICT = "Predict"
    DATASET = "Dataset"

    def run(self):
        if self == Task.LEARN:
            Task.learn()
        elif self == Task.PREDICT:
            Task.predict()
        elif self == Task.DATASET:
            Task.dataset()

    @staticmethod
    def learn():
        name = streamlit.text_input("Model name", value="cnn")

        num_layers = streamlit.select_slider(label="Number of hidden layers", options=[1, 2, 3, 4])
        channels = [8] * num_layers
        kernels = [2] * num_layers

        cols = streamlit.columns(num_layers)
        for i in range(num_layers):
            channels[i] = cols[i].number_input(
                label=f"Channel for Conv layer#{i}",
                min_value=2,
                value=8,
            )
            kernels[i] = cols[i].number_input(
                label=f"Kernel size for Conv layer#{i}",
                min_value=2,
                value=2,
            )

        graph = graphviz.Digraph(
            graph_attr={
                "rankdir": "TB",
            },
            node_attr={
                "shape": "box",
            },
        )
        last = "INPUT (28x28)"
        width = 28
        for i in range(num_layers):
            node = f"L{i}"
            graph.edge(last, node)
            graph.node(node, f"Conv (ch={channels[i]}, ker={kernels[i]}) + Pooling")
            width = (width - kernels[i] + 1) // 2
            last = node
        size = width * width * channels[-1]
        graph.edge(last, "FLATTEN")
        graph.node("FLATTEN", f"Flatten (size={size})")
        graph.edge("FLATTEN", "OUTPUT (10)")
        streamlit.graphviz_chart(graph)

        epochs = streamlit.number_input("Epochs", min_value=1, value=5, max_value=10)

        if streamlit.button("Learn"):
            params = {
                "name": name,
                "hyperparams": {
                    "model_type": "cnn",
                    "channels": channels,
                    "kernels": kernels,
                    "epochs": epochs,
                },
            }
            res = API.post("/api/learn", params)
            streamlit.json(res.json())

    @staticmethod
    def predict():
        models = API.get("/api/models").json()
        name = streamlit.selectbox(label="name", options=list(m["name"] for m in models))
        last_version = next(m["version"] for m in models if m["name"] == name)
        version = streamlit.number_input(
            label="version", min_value=1, max_value=last_version, value=last_version
        )

        canvas = st_canvas(
            stroke_width=13,
            stroke_color="black",
            background_color="white",
            background_image=None,
            update_streamlit=True,
            height=280,
            width=280,  # x10 size
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas.image_data is not None and streamlit.button("Predict!"):
            img = cv2.resize(
                canvas.image_data.astype("uint8"), (28, 28), interpolation=cv2.INTER_LINEAR
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            params = {
                "name": name,
                "version": version,
                "data": img.tolist(),
            }
            res = API.post("/api/predict", params).json()
            streamlit.markdown(f"- **Predicted:** {res['result']}")
            streamlit.bar_chart(res["prob"])

    @staticmethod
    def dataset():
        stat = API.get("/api/dataset").json()
        streamlit.write(stat)


def main():
    streamlit.title("sample-MNIST")

    taskname = streamlit.sidebar.selectbox(
        label="What to do?", options=[task.value for task in Task]
    )
    streamlit.sidebar.markdown("---")

    streamlit.subheader(taskname)
    Task(taskname).run()


if __name__ == "__main__":
    main()
