from glob import glob
import pandas as pd
import re
import streamlit as st
from streamlit_plotly_events import plotly_events
from copy import deepcopy
import utils


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


section_topics_paths = glob("app/data/topic_probs/*.csv")
sections = []
for ix, section_topic_path in enumerate(section_topics_paths):
    sections.append(re.search("([A-Z]|[a-z]|_)+_probs.csv", section_topic_path).group()[:-10])
    sections[ix] = sections[ix].replace("_", " ")

st.title("Clusters visualization")

selected_section = st.selectbox("Select section", sections, index=0)
selected_section_path = "app/data/topic_probs/" + selected_section.replace(" ", "_") + "_probs.csv"

topics_load = load_df_data(selected_section_path)
topics = deepcopy(topics_load)

clusterings = ["Hierarchical", "K-Means", "HDBSCAN"]

col_cluster, col_params = st.columns(2)
with col_cluster:
    selected_clustering = st.selectbox("Select clustering:", clusterings, index = 0)
with col_params:
    if selected_clustering == "Hierarchical":
        t = st.number_input(f"Select t", min_value=0.0, value=1.0, step=1e-5, format="%.5f")
        linkage = utils.calculate_linkage_matrix(topics.iloc[:, 1:-2])
        labels = utils.get_hierarchical_clusters(linkage, t)
    elif selected_clustering == "K-Means":
        n_clusters = st.number_input(f"Select n_clusters", min_value=2, value=2)
        labels = utils.get_kmeans_clusters(topics.iloc[:, 1:-2], n_clusters)
    elif selected_clustering == "HDBSCAN":
        min_cluster_size = st.number_input(f"Select min_cluster_size", min_value=1, value=5)
        min_samples = st.number_input(f"Select min_samples", min_value=1, value=1)
        cluster_selection_epsilon = st.number_input(f"Select cluster_selection_epsilon", min_value=0.0, value=0.0, step=1e-5, format="%.5f")
        labels = utils.get_hdbscan_clusters(topics.iloc[:, 1:-2], min_cluster_size, min_samples, cluster_selection_epsilon)

topics["label"] = labels.astype('int64')

fig = utils.plot_clusters(topics)
selected_point = plotly_events(fig)
if selected_point != []:
    selected_country = topics["country"].loc[(topics["c1"] == selected_point[0]['x']) & (topics["c2"] == selected_point[0]['y'])].values[0]
    st.subheader(f"Country details: {selected_country}")
    st.write("Topic distances:")
    print(topics, selected_country)
    st.plotly_chart(utils.plot_topic_distances(topics, selected_country))
    st.write("Topic distribution: ")
    st.pyplot(utils.plot_topic_distribution(topics, selected_country))
