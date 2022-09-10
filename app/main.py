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
clusterings = ["Hierarchical", "K-Means", "HDBSCAN"]

metric_choices = {"ir": "information radius", "hd": "Hellinger distance"}


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_df_keywords_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


st.title("Topic Modelling for NECPs")

with st.sidebar:
    selected_section = st.selectbox("Select section", sections, index=0)
    version, mapping = st.columns(2)
    with version:
        selected_version = st.selectbox(
            "Select version", versions, index=0, help="Choose version of topic modeling "
        )

    selected_section_path = (
        DIR + str(selected_version) + "_" + selected_section.replace(" ", "_") + "_probs.csv"
    )
    topics_load = load_df_data(selected_section_path)
    keywords_load = load_df_keywords_data(selected_section_path.replace("probs", "topic_words"))
    topics = deepcopy(topics_load)
    keywords = deepcopy(keywords_load)
    is_multidimensional = False
    if selected_section in [
        "Impact Assessment of Planned Policies and Measures",
        "Overview and Process for Establishing the Plan",
    ]:
        n_topics = len(topics.columns[1:-4])
        topic_names = topics.columns[1:-4]
        topic_matrix = topics.iloc[:, 1:-4].values
    else:
        is_multidimensional = True
        n_topics = int(len(topics.columns[1:-4]) / 3)
        topic_names = topics.columns[1:-4][:n_topics]
        topic_matrix = topics.iloc[:, 1:-4].values / 3

    with mapping:
        selected_mapping = st.selectbox(
            "Select mapping",
            mappings,
            index=0,
            help="Choose visualization method for representation of countries on 2D plot based on their agendas",
        )

    selected_clustering = st.selectbox("Select clustering method", clusterings, index=0)

    if selected_clustering == "Hierarchical":
        distance_metric = st.selectbox(
            "Select option", metric_choices.keys(), format_func=lambda x: metric_choices[x], index=0
        )

        linkage_method = st.selectbox(
            "Select linkage algorithm",
            ["average", "single", "complete", "weighted"],
            index=0,
        )

        linkage = utils.calculate_linkage_matrix(topic_matrix, linkage_method, distance_metric)

        t = st.slider(
            f"Select distance threshold",
            min_value=0.0,
            value=float(np.mean(linkage[:, 2])),
            max_value=float(np.max(linkage[:, 2])),
            step=1e-5,
            format="%.5f",
        )
        labels = utils.get_hierarchical_clusters(linkage, t)

    elif selected_clustering == "K-Means":
        n_clusters = st.number_input(f"Select number of clusters", min_value=2, value=2)
        labels = utils.get_kmeans_clusters(topic_matrix, n_clusters)

    elif selected_clustering == "HDBSCAN":
        distance_metric = st.selectbox(
            "Select option", metric_choices.keys(), format_func=lambda x: metric_choices[x], index=0
        )
        min_cluster_size = st.number_input(f"Select minimum cluster size", min_value=1, value=5)
        min_samples = st.number_input(f"Select minimum number of samples", min_value=1, value=1)
        cluster_selection_epsilon = st.number_input(
            f"Select distance threshold",
            min_value=0.0,
            value=0.0,
            step=1e-5,
            format="%.5f",
        )

        distance_matrix = utils.calculate_distance_matrix(
            pd.DataFrame(topic_matrix), distance_metric
        )

        labels = utils.get_hdbscan_clusters(
            distance_matrix,
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

tabs = st.tabs(["Country details", "Topic details"])
with tabs[0]:
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

with tabs[1]:
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
