import streamlit as st
import pandas as pd
import requests
from github import Github
import io
import base64

# Assuming you have a GitHub token stored securely, not hardcoded
g = Github("GHSAT0AAAAAACUFGUDHWTOAQOLVP23Z67C2ZW4ZJSQ")

repo = g.get_repo("No-Country-simulation/-c20-63-m-data-bi")
contents = repo.get_contents("main/carrito.csv")

# Decode the content if it's base64 encoded

file_content = base64.b64decode(contents.content).decode()
df = pd.read_csv(io.StringIO(file_content))

st.title("Hello DS")

st.info("This app builds a ML model")
df





