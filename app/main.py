import pandas as pd
import numpy as np
import streamlit as st
from copy import deepcopy
import utils

import config


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_df_mapping_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_df_keywords_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


st.title("Policy Comparison App")
with st.sidebar:
    selected_section = st.selectbox("Select section", config.SECTIONS, index=0)
    topics_load = utils.convert_country(
        load_df_data(config.SETTINGS_DICT["sections"][selected_section]["probs"])
    )
    mapping_load = utils.convert_country(
        load_df_mapping_data(config.SETTINGS_DICT["sections"][selected_section]["mapping"])
    )
    keywords_load = load_df_keywords_data(config.SETTINGS_DICT["sections"][selected_section]["topic_words"])

    topics = deepcopy(topics_load)
    mapping = deepcopy(mapping_load)
    keywords = deepcopy(keywords_load)
    n_topics = len(topics.columns[1:])
    topic_names = topics.columns[1:]
    topic_matrix = topics.iloc[:, 1:].values

    selected_mapping = st.selectbox(
        "Select mapping",
        config.MAPPINGS,
        index=0,
        help="Choose visualization method for representation of countries on 2D plot based on their agendas",
    )

    selected_clustering = st.selectbox("Select clustering method", config.CLUSTERINGS, index=0)

    if selected_clustering == "Hierarchical":
        distance_metric = st.selectbox(
            "Select option",
            config.METRIC_CHOICES.keys(),
            format_func=lambda x: config.METRIC_CHOICES[x],
            index=0,
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
            value=float(np.median(linkage[:, 2])),
            max_value=float(np.max(linkage[:, 2])),
            step=1e-5,
            format="%.5f",
        )
        labels = utils.get_hierarchical_clusters(linkage, t).astype(str)

    elif selected_clustering == "K-Means":
        n_clusters = st.number_input(f"Select number of clusters", min_value=2, value=4)
        labels = utils.get_kmeans_clusters(topic_matrix, n_clusters).astype(str)

    elif selected_clustering == "HDBSCAN":
        distance_metric = st.selectbox(
            "Select option",
            config.METRIC_CHOICES.keys(),
            format_func=lambda x: config.METRIC_CHOICES[x],
            index=0,
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

        distance_matrix = utils.calculate_distance_matrix(pd.DataFrame(topic_matrix), distance_metric)

        labels = utils.get_hdbscan_clusters(
            distance_matrix,
            min_cluster_size,
            min_samples,
            cluster_selection_epsilon,
        ).astype(str)

x, y = ("c1", "c2") if selected_mapping == "tSNE" else ("u1", "u2")
st.plotly_chart(
    utils.plot_clusters(topics, mapping, labels, x, y),
    config=config.DEFAULT_CONFIG,
)

sc1, sc2 = st.columns(2)
with sc1:
    st.metric("Number of clusters", len(np.unique(labels)))
with sc2:
    pval = utils.manova_significant_difference_pval(topics.iloc[:, 1:], labels)
    if pval < 1e-6:
        pval = "<1e-6"
        st.metric("MANOVA p value", pval)
    else:
        st.metric(
            "MANOVA p value",
            "{:.6f}".format(round(pval, 6)),
        )

tabs = st.tabs(["Country details", "Topic details", "Additional data comparision"])
with tabs[0]:
    selected_country = st.selectbox("Select country", topics.country.unique(), index=0)
    st.header(f"Country details: {selected_country}")

    st.plotly_chart(
        utils.plot_topic_distribution_radar(topics, selected_country),
        config=config.DEFAULT_CONFIG,
    )
    st.plotly_chart(
        utils.plot_topic_distribution_violinplot(topics, selected_country),
        config=config.DEFAULT_CONFIG,
    )

with tabs[1]:
    st.header("Topic analysis")
    with open(config.SETTINGS_DICT["sections"][selected_section]["vis"], "r") as file:
        html_string = file.read()
    st.components.v1.html(html_string, width=800, height=800, scrolling=True)
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
        # topic_num = order_dict[selected_section][i] - 1
        topic_num = i  # TODO to generalize
        st.pyplot(utils.plot_topics(keywords, i, topic_num, topic_names[topic_num], colors_list[topic_num]))

with tabs[2]:
    if len(config.SETTINGS_DICT["additional_files"]) > 0:
        topics_additional = topics.copy()
        # TODO to generalize
        # topics_additional["country"] = topics_additional["country"]
        # selected_columns = st.multiselect(
        #     "Select topic modelling columns",
        #     list(topics_additional.columns[1:-1]),
        #     default=list(topics_additional.columns[1:-1]),
        # selected_data = st.selectbox(
        #     "Select additional data",
        #     ["Comissions Individual Assesment", "Planning for net zero report"],
        #     index=1,
        # )
        # df_selected = (
        #     df_comissions_individual_assesment.copy()
        #     if selected_data == "Comissions Individual Assesment"
        #     else df_planning_for_net_zero_report.copy()
        # )
        # df_selected["country"] = df_selected["country"].apply(lambda x: x.lower())
        # default = (
        #     [colname for colname in df_selected.columns if "total" in colname.lower()]
        #     if selected_data == "Planning for net zero report"
        #     else list(df_selected.columns[1:])[:4]
        # )
        # selected_columns_additional = st.multiselect(
        #     "Select additional data columns",
        #     list(df_selected.columns[1:]),
        #     default=default,
        # )
        # merged_df = topics_additional[selected_columns + ["country"]].merge(
        #     df_selected[selected_columns_additional + ["country"]], how="left", on="country"
        # )
        # selected_method = st.selectbox(
        #     "Select correlation method",
        #     ["Pearson", "Kendall", "Spearman"],
        #     index=0,
        # )
        # corr_df = merged_df.corr(method=selected_method.lower())
        # corr_df = corr_df.drop(selected_columns, axis=1)
        # corr_df = corr_df.drop(selected_columns_additional, axis=0)
        # st.header("Correlation heatmap")
        # st.write(
        #     utils.plot_correlation_heatmap(corr_df),
        #     config=config.DEFAULT_CONFIG,
        # )
    else:
        st.write("There aren't any additional files defined")
