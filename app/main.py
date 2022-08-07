from glob import glob

import pandas as pd
import re
import streamlit as st
from streamlit_plotly_events import plotly_events

from plots.clustering import plot_clusters
from plots.clustering import plot_topic_distribution
from plots.clustering import plot_topic_distances


cluster_files_paths = glob("app/data/clustering_files/*.csv")
cluster_files = []
for ix, cluster_file in enumerate(cluster_files_paths):
    cluster_files.append(re.search("([A-Z]|[a-z]|_)+_hierarchical.csv", cluster_file).group()[:-17])
    cluster_files[ix] = cluster_files[ix].replace("_", " ")


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


st.title("Clusters visualization")

cluster_filename = st.selectbox("Select a clustering", cluster_files, index=len(cluster_files)-1)
cluster_filename = cluster_filename.replace(" ", "_")
cluster_filename = "app/data/clustering_files/" + cluster_filename +"_hierarchical.csv"

clusters = load_df_data(cluster_filename)
tsne_filename = cluster_filename[26:-16] + "tsne.csv"
tsne = load_df_data(str("app/data/tsne_files/" + tsne_filename))
df = pd.merge(tsne, clusters, on="country").astype({"label":str})

fig = plot_clusters(df)
selected_point = plotly_events(fig)
if selected_point != []:
    selected_country = df["country"].loc[(df["c1"] == selected_point[0]['x']) & (df["c2"] == selected_point[0]['y'])].values[0]
    st.subheader(f"Country details: {selected_country}")
    st.write("Topic distances:")
    print(df, selected_country)
    st.plotly_chart(plot_topic_distances(df, selected_country))
    st.write("Topic distribution: ")
    st.pyplot(plot_topic_distribution(df, selected_country))
