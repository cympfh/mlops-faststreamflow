import streamlit
from main import API

streamlit.title("sample-MNIST/Dataset 📚")

stat = API.get("/api/dataset").json()
streamlit.write(stat)
