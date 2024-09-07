import streamlit as st
import pandas as pd
import requests
from github import Github
import io


# Assuming you have a GitHub token stored securely, not hardcoded
g = Github("GHSAT0AAAAAACUFGUDHWTOAQOLVP23Z67C2ZW4ZJSQ")

repo = g.get_repo("No-Country-simulation/-c20-63-m-data-bi")
contents = repo.get_contents("main/carrito.csv")

# Decode the content if it's base64 encoded
import base64
file_content = base64.b64decode(contents.content).decode()

st.title("Hello DS")

st.info("This app builds a ML model")

df = pd.read_csv(io.StringIO(file_content))
df





