import cv2
import streamlit
from main import API
from streamlit_drawable_canvas import st_canvas

streamlit.title("sample-MNIST/Predict 🔬")

models = API.get("/api/models").json()
if len(models) == 0:
    streamlit.markdown("No models learned yet. [Learn](learn/) your model!")
    streamlit.stop()

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
