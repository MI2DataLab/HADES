import pandas as pd
import numpy as np
import streamlit as st
from copy import deepcopy
import utils
import json
import spacy

import config_resilience_plans as config


@st.cache
def load_df_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_df_mapping_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_df_keywords_data(df_file_name: str, index_col=False) -> pd.DataFrame:
    return pd.read_csv(df_file_name, index_col=index_col)


@st.cache
def load_additional_dfs() -> dict:
    return {
        path.split("/")[-1]:pd.read_csv(path, index_col=0)
        for path in config.SETTINGS_DICT["additional_files"]
    }


@st.cache
def load_summaries(file_path: str) -> json:
    f = open(file_path)
    j = json.load(f)
    f.close()
    return j


@st.cache
def load_essentials(file_path: str) -> json:
    f = open(file_path)
    j = json.load(f)
    f.close()
    return j


st.markdown(
    f"""
    <style>
        .streamlit-expanderHeader{{
            font-size: calc(1.2rem + 1.2vw);
            font-family: "Source Sans Pro", sans-serif;
            font-weight: 600;
            color: rgb(22, 14, 59);
            letter-spacing: -0.005em;
            line-height: 1.2;
        }}
        .css-12oz5g7{{
            max-width: 100%;
            padding: 5%;
        }}
        .css-vxbmln{{
            width: 244px;
        }}
        .imp_word{{
            color: blue;
        }}
        .css-1bn4mii{{
            transform: scale(0.93);
            margin-left: -50px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if 'en' not in st.session_state:
        st.session_state.en =  spacy.load('en_core_web_sm')

st.title("Policy Comparison App")

with st.sidebar:
    selected_section = st.selectbox("Select section", config.SECTIONS, index=0)
    topics_load = load_df_data(config.SETTINGS_DICT["sections"][selected_section]["probs"])
    mapping_load = load_df_mapping_data(config.SETTINGS_DICT["sections"][selected_section]["mapping"])
    keywords_load = load_df_keywords_data(
        config.SETTINGS_DICT["sections"][selected_section]["topic_words"]
    )
    essentials_load = load_essentials(
        config.SETTINGS_DICT["sections"][selected_section]["essentials"]
    )
    summaries_load = load_summaries(config.SETTINGS_DICT["summaries_file"])

    topics = deepcopy(topics_load)
    mapping = deepcopy(mapping_load)
    keywords = deepcopy(keywords_load)
    essential_sentences = deepcopy(essentials_load)
    summaries = deepcopy(summaries_load)
    n_topics = len(topics.columns[1:])
    topic_names = topics.columns[1:]
    topic_matrix = topics.iloc[:, 1:].values

    selected_mapping = st.selectbox(
        "Select mapping",
        config.MAPPINGS,
        index=0,
        help="Method for representation of documents on 2D plot based on their contents",
    )

    selected_clustering = st.selectbox(
        "Select clustering method",
        config.CLUSTERINGS,
        index=0,
        help="Method used for grouping documents",
    )

    if selected_clustering == "Hierarchical":
        distance_metric = st.selectbox(
            "Select metric",
            config.METRIC_CHOICES.keys(),
            format_func=lambda x: config.METRIC_CHOICES[x],
            index=0,
            help="Metric for calculating distances between contents",
        )

        linkage_method = st.selectbox(
            "Select linkage algorithm",
            ["average", "single", "complete", "weighted"],
            index=0,
            help="Linkage scheme used for grouping documents",
        )

        linkage = utils.calculate_linkage_matrix(
            topic_matrix,
            linkage_method,
            distance_metric,
        )

        t = st.slider(
            f"Select distance threshold",
            min_value=0.0,
            value=float(np.median(linkage[:, 2])),
            max_value=float(np.max(linkage[:, 2])),
            step=1e-5,
            format="%.5f",
            help="Distance threshold used for creating groups",
        )
        labels = utils.get_hierarchical_clusters(linkage, t).astype(str)

    elif selected_clustering == "K-Means":
        n_clusters = st.number_input(
            f"Select number of clusters",
            min_value=2,
            value=4,
        )
        labels = utils.get_kmeans_clusters(topic_matrix, n_clusters).astype(str)

    elif selected_clustering == "HDBSCAN":
        distance_metric = st.selectbox(
            "Select option",
            config.METRIC_CHOICES.keys(),
            format_func=lambda x: config.METRIC_CHOICES[x],
            index=0,
        )
        min_cluster_size = st.number_input(
            f"Select minimum cluster size",
            min_value=1,
            value=5,
        )
        min_samples = st.number_input(
            f"Select minimum number of samples",
            min_value=1,
            value=1,
        )
        cluster_selection_epsilon = st.number_input(
            f"Select distance threshold",
            min_value=0.0,
            value=0.0,
            step=1e-5,
            format="%.5f",
        )

        distance_matrix = utils.calculate_distance_matrix(
            pd.DataFrame(topic_matrix),
            distance_metric,
        )

        labels = utils.get_hdbscan_clusters(
            distance_matrix,
            min_cluster_size,
            min_samples,
            cluster_selection_epsilon,
        ).astype(str)

x, y = ("c1", "c2") if selected_mapping == "tSNE" else ("u1", "u2")

sc1, sc2 = st.columns([4, 1])
with sc1:
    if config.COUNTRIES_DIVISION:
        map_tab, clustering_tab = st.tabs(["Map", "Clustering"])
        with map_tab:
            st.plotly_chart(
                utils.plot_map(topics, mapping, labels),
                config=config.DEFAULT_CONFIG,
                use_container_width=True,
            )
        with clustering_tab:
            st.plotly_chart(
                utils.plot_clusters(topics, mapping, labels, x, y),
                config=config.DEFAULT_CONFIG,
                use_container_width=True,
            )
    else:
        st.plotly_chart(
                utils.plot_clusters(topics, mapping, labels, x, y, try_flags=False, text=config.DIVISION_COLUMN),
                config=config.DEFAULT_CONFIG,
                use_container_width=True,
        )
with sc2:
    st.markdown("#"); st.markdown("#"); st.markdown("#"); st.markdown("#"); st.markdown("#"); st.markdown("#")
    st.metric("Number of clusters", len(np.unique(labels)))
    pval = utils.manova_significant_difference_pval(
        topics.iloc[:, 1:], labels
    )
    if pval < 1e-6:
        pval = "<1e-6"
        st.metric("MANOVA p value", pval)
    else:
        st.metric(
            "MANOVA p value",
            "{:.6f}".format(round(pval, 6)),
        )
    if config.COUNTRIES_DIVISION:
        st.markdown("#"); st.markdown("#"); st.markdown("#"); st.markdown("#"); st.markdown("#"); st.markdown("#")

tabs = st.tabs(["Document details", "Topic details", "Additional data comparision"])
with tabs[0]:
    selected_document = st.selectbox("Select document", topics[config.DIVISION_COLUMN].unique(), index=0)
    st.header(f"Document details: {selected_document}")

    st.markdown(f"""<h4 style="padding-top: 0px;">Section summary:</h4>""", unsafe_allow_html=True)
    st.write(summaries[selected_section][selected_document])
    st.markdown(f"""<hr style='margin: 0px;'>""", unsafe_allow_html=True)
    # TO DO
    clean_topics = [topic_name.split(" ", 1)[1] for topic_name in topic_names[np.array(config.order_dict[selected_section]) - 1]]
    # clean_topics = [topic_name.split(" ", 1)[1] for topic_name in topic_names]
    selected_topic = st.selectbox(
        "Select topic",
        clean_topics,
        index=0)
    topic_num = 0
    for idx, topic in enumerate(clean_topics):
        if selected_topic == topic:
            topic_num = idx
    st.markdown(f"""<h4 style="padding-top: 0px;">Essential sentences:</h4>""", unsafe_allow_html=True)
    for i in range(3):   
        ess_words = list(essential_sentences[selected_document][str(topic_num)]['words'].keys())
        ess_sentence = essential_sentences[selected_document][str(topic_num)]['sentences'][i][0]
        ess_sentence_splitted = ess_sentence.split()
        html_sentence = "<p>"
        for word in ess_sentence_splitted:
            word_en = st.session_state.en(word)
            is_imp = bool(word_en[0].lemma_ in ess_words)
            if is_imp:
                html_sentence = html_sentence + " <span class='imp_word'>" + word + "</span>"
            else:
                html_sentence = html_sentence + " " + word
        html_sentence = html_sentence + "</p>"
        st.markdown(f"""{html_sentence}""", unsafe_allow_html=True)

    st.header(f"Compare documents")
    selected_entities = st.multiselect(
        label="Select document",
        options=topics[config.DIVISION_COLUMN].unique(),
        default=topics[config.DIVISION_COLUMN].unique()[:2],  # assumption that there are two entities
    )
    radar_col, topic_dist_col = st.columns(2)
    with radar_col:
        # st.markdown("#"); st.markdown("#");
        st.plotly_chart(
            utils.plot_topic_distribution_radar(topics, selected_entities, app_format=True),
            config=config.DEFAULT_CONFIG,
            use_container_width=True,
            width=500,
        )
        #### EXPERIMENTAL ####
        topic_names = np.hstack(topics.columns[1:])
        html_text_legend = "".join(["<span style='display: inline-block;'>T" + str(i) + ": " + topic_names[i] + "  &#x2022; </span>" for i in range(len(topic_names))])
        html_string = f"""
        <div style="border: 1px solid #808495;border-radius: 5px;padding: 10px;margin: 5px 20px; color: #808495;">
            {html_text_legend}
        </div>
        """
        st.markdown(html_string, unsafe_allow_html=True)
        #### EXPERIMENTAL ####
    with topic_dist_col:
        st.plotly_chart(
            utils.plot_topic_distribution_violinplot(topics, selected_entities),
            config=config.DEFAULT_CONFIG,
            use_container_width=True,
            width=500,
        )

with tabs[1]:
    st.header("Topic analysis")
    with open(config.SETTINGS_DICT["sections"][selected_section]["vis"], "r") as file:
        html_string = file.read()
    st.components.v1.html(
        html_string,
        width=1250,
        height=800,
        scrolling=False
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
    keywords_col1, keywords_col2 = st.columns(2)
    for i in range(n_topics):
        topic_num = config.order_dict[selected_section][i] - 1
        # topic_num = i  # TODO to generalize
        if i % 2:
            keywords_col2.pyplot(
                utils.plot_topics(
                    keywords,
                    i,
                    topic_num,
                    topic_names[topic_num],
                    colors_list[topic_num],
                )
            )
        else:
            keywords_col1.pyplot(
                utils.plot_topics(
                    keywords,
                    i,
                    topic_num,
                    topic_names[topic_num],
                    colors_list[topic_num],
                )
            )
    

with tabs[2]:
    if len(config.SETTINGS_DICT["additional_files"]) > 0:
        st.header("Correlation heatmap")
        heatmap_params_col, heatmap_plot_col = st.columns(2)
        with heatmap_params_col:
            topics_additional = topics.copy()
            topics_additional[config.DIVISION_COLUMN] = topics_additional[config.DIVISION_COLUMN].apply(
                lambda x: x.lower()
            )
            num_topics = len(topics_additional.columns)-1
            selected_columns = st.multiselect(
                "Select topic modelling columns",
                list(topics_additional.columns[1:(num_topics+1)]),
                default=list(topics_additional.columns[1:(num_topics+1)]), 
            )
            additional_dfs = load_additional_dfs()
            selected_data = st.selectbox(
                "Select additional data",
                list(additional_dfs.keys()),
                index=0,
            )
            df_selected = deepcopy(additional_dfs[selected_data])
            df_selected[config.DIVISION_COLUMN] = df_selected[config.DIVISION_COLUMN].apply(lambda x: x.lower())
            default = list(df_selected.columns[1:])[0]
            
            selected_columns_additional = st.multiselect(
                "Select additional data columns",
                list(df_selected.columns[1:]),
                default=default,
            )
            merged_df = topics_additional[selected_columns + [config.DIVISION_COLUMN]].merge(
                df_selected[selected_columns_additional + [config.DIVISION_COLUMN]],
                how="left",
                on=config.DIVISION_COLUMN,
            )
            selected_method = st.selectbox(
                "Select correlation method",
                ["Pearson", "Kendall", "Spearman"],
                index=0,
            )
            corr_df = merged_df.corr(method=selected_method.lower())
            corr_df = corr_df.drop(selected_columns, axis=1)
            corr_df = corr_df.drop(selected_columns_additional, axis=0)
        with heatmap_plot_col:
            st.markdown("#")
            st.markdown("#")
            st.pyplot(utils.plot_correlation_heatmap(corr_df))
    else:
        st.write("There aren't any additional files defined")
