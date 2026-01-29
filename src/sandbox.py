import plotly.graph_objects as go
import time
import streamlit as st

if st.button("TEST: kaleido"):
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1])
    t0 = time.perf_counter()
    png = fig.to_image(format="png", width=300, height=200, scale=1)
    st.write("ok seconds:", time.perf_counter()-t0, "bytes:", len(png))
    st.download_button("download test png", data=png, file_name="kaleido_test.png", mime="image/png")
