import streamlit as st
import pandas as pd
import requests

# Your GitHub Personal Access Token (PAT)
token = "GHSAT0AAAAAACUFGUDHWTOAQOLVP23Z67C2ZW4ZJSQ"

# URL to the raw file in the private repository
url = "https://raw.githubusercontent.com/No-Country-simulation/-c20-63-m-data-bi/main/carrito.csv"

# Set up headers for authentication
headers = {
    "Authorization": f"token {token}"
}

# Fetch the CSV file
response = requests.get(url, headers=headers)

st.title("Hello DS")

st.info("This app builds a ML model")

from io import StringIO
csv_data = StringIO(response.text)
df = pd.read_csv(csv_data)
df