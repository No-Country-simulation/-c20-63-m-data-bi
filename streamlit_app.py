import streamlit as st
import pandas as pd
st.title("Hello DS")

st.info("This app builds a ML model")
# Get the raw GitHub URL for the file
url = "https://raw.githubusercontent.com/No-Country-simulation/-c20-63-m-data-bi/main/carrito.csv?token=GHSAT0AAAAAACUFGUDGAY2LLEBLLJEE74FMZW4Y6TA"

# Read the Excel file into a pandas DataFrame
df = pd.read_csv(url)
df
