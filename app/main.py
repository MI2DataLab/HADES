from glob import glob
import pandas as pd
import numpy as np
import re
import streamlit as st
from copy import deepcopy
import utils

DIR = "app/data/"

default_config = {
    "displaylogo": False,
    "staticPlot": False,
    "toImageButtonOptions": {
        "height": None,
        "width": None,
    },
    "modeBarButtonsToRemove": [
        "sendDataToCloud",
        "lasso2d",
        "autoScale2d",
        "select2d",
        "zoom2d",
        "pan2d",
        "zoomIn2d",
        "zoomOut2d",
        "resetScale2d",
        "toggleSpikelines",
        "hoverCompareCartesian",
        "hoverClosestCartesian",
    ],
}


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_df_keywords_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


section_topics_paths = glob(DIR + "*probs.csv")
sections = [
    "Decarbonisation",
    "Energy efficiency",
    "Energy security",
    "Internal market",
    "R&I and Competitiveness",
    "Impact Assessment of Planned Policies and Measures",
    "Overview and Process for Establishing the Plan",
]

versions = [50, 100, 150, 200, 250]
mappings = ["tSNE", "UMAP"]

st.title("Topic Modelling for NECPs")

selected_section = st.selectbox("Select section", sections, index=0)

version, mapping = st.columns(2)
with version:
    selected_version = st.selectbox(
        "Select version", versions, index=0, help="Choose version of topic modeling "
    )
with mapping:
    selected_mapping = st.selectbox(
        "Select mapping",
        mappings,
        index=0,
        help="Choose visualization method for representation of countries on 2D plot based on their agendas",
    )

selected_section_path = (
    DIR + str(selected_version) + "_" + selected_section.replace(" ", "_") + "_probs.csv"
)

topics_load = load_df_data(selected_section_path)
keywords_load = load_df_keywords_data(selected_section_path.replace("probs", "topic_words"))
topics = deepcopy(topics_load)
keywords = deepcopy(keywords_load)
if selected_section in [
    "Impact Assessment of Planned Policies and Measures",
    "Overview and Process for Establishing the Plan",
]:
    n_topics = len(topics.columns[1:-4])
    topic_names = topics.columns[1:-4]
else:
    n_topics = int(len(topics.columns[1:-4]) / 3)
    topic_names = topics.columns[1:-4][:n_topics]

clusterings = ["Hierarchical", "K-Means", "HDBSCAN"]
col_cluster, col_metric, col_params = st.columns(3)
with col_cluster:
    selected_clustering = st.selectbox("Select clustering:", clusterings, index=0)
with col_params:
    if selected_clustering == "Hierarchical":
        linkage = utils.calculate_linkage_matrix(topics.iloc[:, 1:-4].values / 3)
        t = st.slider(
            f"Select t",
            min_value=0.0,
            value=float(np.mean(linkage[:, 2])),
            max_value=float(np.max(linkage[:, 2])),
            step=1e-5,
            format="%.5f",
        )
        labels = utils.get_hierarchical_clusters(linkage, t)
    elif selected_clustering == "K-Means":
        n_clusters = st.number_input(f"Select n_clusters", min_value=2, value=2)
        labels = utils.get_kmeans_clusters(topics.iloc[:, 1:-4].values / 3, n_clusters)
    elif selected_clustering == "HDBSCAN":
        min_cluster_size = st.number_input(f"Select min_cluster_size", min_value=1, value=5)
        min_samples = st.number_input(f"Select min_samples", min_value=1, value=1)
        cluster_selection_epsilon = st.number_input(
            f"Select cluster_selection_epsilon",
            min_value=0.0,
            value=0.0,
            step=1e-5,
            format="%.5f",
        )
        labels = utils.get_hdbscan_clusters(
            topics.iloc[:, 1:-4].values / 3,
            min_cluster_size,
            min_samples,
            cluster_selection_epsilon,
        )

topics["label"] = labels.astype("str")

x, y = ("c1", "c2") if selected_mapping == "tSNE" else ("u1", "u2")
st.plotly_chart(
    utils.plot_clusters(topics, x, y),
    config=default_config,
)

sc1, sc2 = st.columns(2)
with sc1:
    st.metric("Number of clusters", len(np.unique(labels)))
with sc2:
    pval = utils.manova_significant_difference_pval(topics.iloc[:, 1:-5], topics["label"])
    if pval < 1e-6:
        pval = "<1e-6"
        st.metric("MANOVA p value", pval)
    else:
        st.metric(
            "MANOVA p value",
            "{:.6f}".format(round(pval, 6)),
        )

selected_country = st.selectbox("Select country", topics.country.unique(), index=0)
st.header(f"Country details: {selected_country}")

if selected_section in [
    "Impact Assessment of Planned Policies and Measures",
    "Overview and Process for Establishing the Plan",
]:
    st.plotly_chart(
        utils.plot_topic_distribution_radar(topics, selected_country),
        config=default_config,
    )
    st.plotly_chart(
        utils.plot_topic_distribution_violinplot(topics, selected_country),
        config=default_config,
    )
else:
    st.subheader("National Objectives and Targets")
    st.plotly_chart(
        utils.plot_topic_distribution_radar(
            topics, selected_country, ind_from=1, ind_to=n_topics + 1
        ),
        config=default_config,
    )
    st.plotly_chart(
        utils.plot_topic_distribution_violinplot(
            topics, selected_country, ind_from=1, ind_to=n_topics + 1
        ),
        config=default_config,
    )
    st.subheader("Policies and Measures")
    st.plotly_chart(
        utils.plot_topic_distribution_radar(
            topics, selected_country, ind_from=n_topics + 1, ind_to=2 * n_topics + 1
        ),
        config=default_config,
    )
    st.plotly_chart(
        utils.plot_topic_distribution_violinplot(
            topics, selected_country, ind_from=n_topics + 1, ind_to=2 * n_topics + 1
        ),
        config=default_config,
    )
    st.subheader("Current Situation and Reference Projections")
    st.plotly_chart(
        utils.plot_topic_distribution_radar(
            topics, selected_country, ind_from=2 * n_topics + 1, ind_to=3 * n_topics + 1
        ),
        config=default_config,
    )
    st.plotly_chart(
        utils.plot_topic_distribution_violinplot(
            topics, selected_country, ind_from=2 * n_topics + 1, ind_to=3 * n_topics + 1
        ),
        config=default_config,
    )


st.header(f"Topic keywords")
colors_list = [
    "#8bdcbe",
    "#f05a71",
    "#371ea3",
    "#46bac2",
    "#ae2c87",
    "#ffa58c",
    "#4378bf",
] * n_topics
for i in range(n_topics):
    st.pyplot(utils.plot_topics(keywords, i, topic_names[i], colors_list[i]))
