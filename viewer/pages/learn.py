import graphviz
import streamlit
from main import API

streamlit.title("sample-MNIST/Learn 🌸")

name = streamlit.text_input("Model name", value="cnn")

num_layers = streamlit.select_slider(
    label="Number of hidden layers", options=[1, 2, 3, 4]
)
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
        min_value=1,
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

epochs = streamlit.number_input("Epochs", min_value=1, value=5)

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
    streamlit.markdown(
        "Visit [mlflow dashboard](http://localhost:6002/) to check learning processes."
    )
