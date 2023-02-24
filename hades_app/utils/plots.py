import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pycountry
import seaborn as sns
from copy import deepcopy
from matplotlib import colors


def plot_map(
    df_probs,
    df_mapping,
    labels,
    map_scope="europe",
    lonaxis_range=[-30, 25],
    lataxis_range=[50, 95],
):
    df_probs["country"] = df_probs["country"].astype(str)
    df_mapping["country"] = df_mapping["country"].astype(str)
    df = df_probs.merge(df_mapping, on="country")
    df["label"] = labels
    df_to_plot = deepcopy(df)
    iso_alpha3 = df_to_plot["country"].apply(
        lambda country: pycountry.countries.search_fuzzy(country.split("_")[-1])[
            0
        ].alpha_3
    )
    colnames = df.columns[:-5].to_list()
    df_to_plot["iso"] = iso_alpha3
    fig = px.choropleth(
        df_to_plot,
        locations="iso",
        scope=map_scope,
        color="label",
        hover_name="country",
        hover_data=colnames,
        projection="azimuthal equal area",
    )
    topic_dist_str = [
        str(colnames[i]) + f": %{{customdata[{i}]:.3f}}"
        for i in range(1, len(colnames))
    ]
    fig.update_traces(
        hovertemplate="<br>".join(
            ["<b>%{customdata[0]}</b>", "<i>Topic distribution:</i>"] + topic_dist_str
        ),
    )
    fig.update_geos(
        fitbounds=False,
        visible=True,
        resolution=50,
        lonaxis_range=lonaxis_range,
        lataxis_range=lataxis_range,
        center=dict(lon=10, lat=50),
    )

    fig.update_layout(
        height=500,
        width=1000,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        dragmode=False,
    )
    return fig


def plot_clusters(
    df_probs,
    df_mapping,
    labels,
    x="c1",
    y="c2",
    entity_colname="country",
    try_flags=True,
    text = None
):
    df_probs[entity_colname] = df_probs[entity_colname].astype(str)
    df_mapping[entity_colname] = df_mapping[entity_colname].astype(str)
    df = df_probs.merge(df_mapping, on=entity_colname)
    df["label"] = labels
    iso_alpha2 = None
    if try_flags:
        iso_alpha2 = df[entity_colname].apply(
            lambda country: pycountry.countries.search_fuzzy(country.split("_")[-1])[0].alpha_2
        )

    minDim = np.mean(np.abs(df[[x, y]].max() - df[[x, y]].min()))
    maxi = 0.07 * minDim

    colnames = df_probs.columns.to_list()

    if try_flags:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color="label",
            hover_data=colnames,
        )
    else:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color="label",
            hover_data=colnames,
            text=text,
        )

    topic_dist_str = [
        str(colnames[i]) + f": %{{customdata[{i}]:.3f}}" for i in range(1, len(colnames))
    ]

    fig.update_traces(
        marker_size=30,
        hovertemplate="<br>".join(
            ["<b>%{customdata[0]}</b>", "<i>Topic distribution:</i>"] + topic_dist_str
        ),
        textfont_color="black",
    )

    if try_flags:
        for i, row in df.iterrows():
            country_iso = iso_alpha2[i]
            fig.add_layout_image(
                dict(
                    source=f"https://raw.githubusercontent.com/matahombres/CSS-Country-Flags-Rounded/master/flags/{country_iso}.png",
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=row[x],
                    y=row[y],
                    sizex=maxi,
                    sizey=maxi,
                    sizing="contain",
                    layer="above",
                )
            )

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)

    return fig


def plot_topic_distances(df, selected_text, entity_colname="country", label="label"):
    topics_number = len(df.columns) - 5
    plot_labels = np.arange(topics_number)
    in_cluster = []
    out_cluster = []

    for idx in range(topics_number):
        cluster = df.loc[df[entity_colname] == selected_text, "label"].values[0]
        in_cluster_texts = df.loc[df[label] == cluster, entity_colname].values

        if len(in_cluster_texts) > 1:
            avg_in = 0
            for text_item in in_cluster_texts:
                avg_in += abs(
                    df.loc[df[entity_colname] == text_item, str(idx)].values[0]
                    - df.loc[df[entity_colname] == selected_text, str(idx)].values[0]
                )
            avg_in /= len(in_cluster_texts) - 1
            in_cluster.append(avg_in)
        else:
            in_cluster.append(0)

        if (len(df[entity_colname].values) - len(in_cluster_texts)) > 0:
            avg_out = 0
            for text_item in df[entity_colname].values:
                if text_item in in_cluster_texts:
                    continue
                avg_out += abs(
                    df.loc[df[entity_colname] == text_item, str(idx)].values[0]
                    - df.loc[df[entity_colname] == selected_text, str(idx)].values[0]
                )
            avg_out /= len(df[entity_colname].values) - len(in_cluster_texts)
            out_cluster.append(avg_out)
        else:
            out_cluster.append(0)

    fig = go.Figure(
        data=[
            go.Bar(name="Inside cluster", x=np.arange(len(plot_labels)), y=in_cluster),
            go.Bar(name="Outside cluster", x=np.arange(len(plot_labels)), y=out_cluster),
        ]
    )

    fig.update_xaxes(type="category")
    fig.update_layout(barmode="group")

    return fig


def plot_topic_distribution(df, selected_text, entity_colname="country", label="label"):
    topics_number = len(df.columns) - 5
    fig, axes = plt.subplots(topics_number, 1, figsize=(15, 1.5 * topics_number))
    plt.subplots_adjust(hspace=0.6)
    fig.suptitle(f"Topics distribution for {selected_text}")
    plt.xlim(0, 1)
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    fig.patch.set_facecolor("#262730")

    for ax in axes:
        ax.tick_params(color="white", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_facecolor("#262730")

    for idx in range(topics_number):
        axes[idx].set_title(f"Topic {idx}")
        axes[idx].axes.yaxis.set_ticklabels([])
        axes[idx].axvline(
            x=df.loc[df[entity_colname] == selected_text, str(idx)].values[0], color="g", lw=3
        )
        axes[idx].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        cluster = df.loc[df[entity_colname] == selected_text, "label"].values[0]
        in_cluster_texts = df.loc[df[label] == cluster, entity_colname].values

        if len(in_cluster_texts) > 1:
            avg_in = df.loc[df[entity_colname].isin(in_cluster_texts), str(idx)].mean()
            axes[idx].axvline(x=avg_in, color="b", lw=3)

        if (len(df[entity_colname].values) - len(in_cluster_texts)) > 0:
            avg_out = df.loc[~df[entity_colname].isin(in_cluster_texts), str(idx)].mean()
            axes[idx].axvline(x=avg_out, color="r", lw=3)

    legend_lines = [
        plt.Line2D([0], [0], color="g", lw=3),
        plt.Line2D([0], [0], color="b", lw=3),
        plt.Line2D([0], [0], color="r", lw=3),
    ]
    fig.legend(
        legend_lines,
        ["selected country", "inside cluster", "outside cluster"],
        facecolor="#262730",
    )

    return fig


def plot_topic_distribution_radar(df, selected_entities, entity_colname="country", app_format=False):
    fig = go.Figure()
    topic_names = np.hstack(df.columns[1:])
    sub_df = df.loc[df[entity_colname].isin(selected_entities)]
    for i, row in sub_df.iterrows():
        r = np.hstack(row[1:].values)
        name = row[0]
        if app_format:
            theta = ["T" + str(i+1) for i in range(len(topic_names))]
        else:
            theta = topic_names
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                name=name,
                text=topic_names,
                hovertemplate="<b>" + name + "</b><br>" + "%{text}: %{r}",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                linecolor="black",
                rangemode="nonnegative",
                tickfont_size=12,
                title_text="Topic probability",
                tickfont_color="black",
                titlefont_color="black",
            ),
            angularaxis=dict(
                rotation=90,
            ),
            bgcolor="rgb(237,237,237)",
        ),
        showlegend=False,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_topic_distribution_violinplot(df, selected_entities, entity_colname="country"):
    df_melted = df.melt(id_vars=entity_colname, value_vars=df.columns[1:])
    selected_ids = np.array(df_melted[df_melted[entity_colname].isin(selected_entities)].index)
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=df_melted["variable"],
            y=df_melted["value"],
            text=df_melted["country"],
            points="all",
            box_visible=True,
            line_color="#46bac2",
            meanline_visible=True,
            pointpos=1.5,
            jitter=0.7,
            selectedpoints=selected_ids,
            selected={"marker_color": "#371ea3", "marker_opacity": 1.0},
            unselected={"marker_opacity": 0.5},
            hovertemplate="<b>%{text}</b>: %{y}<extra>%{x}</extra>",
            hoveron="points",
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Topic",
        yaxis_title="Topic probability",
    )
    return fig


def plot_topics(
    topic_keywords: pd.DataFrame,
    topic_ind: int,
    topic_num: int,
    topic_name: str,
    col: str,
):
    fig, ax1 = plt.subplots()
    ax1.bar(
        x="word",
        height="word_count",
        data=topic_keywords.loc[topic_keywords.topic_id == topic_num, :],
        color=col,
        width=0.5,
        alpha=0.3,
        label="Word Count",
    )
    ax_twin = ax1.twinx()
    ax_twin.bar(
        x="word",
        height="importance",
        data=topic_keywords.loc[topic_keywords.topic_id == topic_num, :],
        color=col,
        width=0.2,
        label="Word Weight",
    )
    ax1.set_ylabel("Word Count", color=col)
    ax_twin.set_ylabel("Word Weight", color=col)

    ax1.set_title(f"Topic {topic_ind + 1}: " + topic_name, color=col, fontsize=12)
    ax1.xaxis.set_ticks(np.arange(len(topic_keywords.loc[topic_keywords.topic_id == topic_num, "word"])))
    ax1.set_xticklabels(
        topic_keywords.loc[topic_keywords.topic_id == topic_num, "word"],
        rotation=30,
        horizontalalignment="right",
        size=8,
    )
    ax1.legend(loc="upper right")
    ax_twin.legend(loc="lower right")
    ax1.grid(False)
    ax_twin.grid(False)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.heatmap(df, ax=ax, annot=True)
    return fig
