from glob import glob

import pandas as pd
import streamlit as st

from plots.clustering import plot_clusters

clusters_files = glob("app/data/clustering_files/*.csv")
tsne_files = glob("app/data/tsne_files/*.csv")


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


st.title("Clusters visualization")

cluster_filename = st.selectbox("Select a clustering", clusters_files)

clusters = load_df_data(cluster_filename)
tsne = load_df_data("app/data/tsne_files/tsne_result.csv")
df = pd.merge(tsne, clusters, on="country").astype({"label":str})

fig = plot_clusters(df)
st.plotly_chart(fig)
