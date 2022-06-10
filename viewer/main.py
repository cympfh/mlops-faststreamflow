import json

import dotenv
import requests
import streamlit


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


streamlit.title("sample-MNIST ❄️")
streamlit.sidebar.title("sample-MNIST ❄️")
streamlit.markdown(
    """
Select a page from sidebar.

- [dataset](dataset/)
    - check dataset stat
- [learn](learn/)
    - Learn a new model
- [predict](predict/)
    - Test a trained model
"""
)
