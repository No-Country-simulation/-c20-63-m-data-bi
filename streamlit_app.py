import streamlit as st
import pandas as pd
import numpy as np


st.title("Hello DS")

st.info("This app builds a ML model")

df = pd.read_csv('carrito.csv')
df