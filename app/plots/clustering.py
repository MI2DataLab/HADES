import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_clusters(df, x="c1", y="c2", color="label", text="country", marker_size=15):
    fig = px.scatter(df, x=x, y=y, color=color, text=text)
    fig.update_layout(
        plot_bgcolor='#262730',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        showlegend=False
    )
    fig.update_traces(marker={'size': marker_size})
    return fig


def plot_topic_distances(df, selected_text, text="country", label="label"):
    topics_number = len(df.columns) - 4
    plot_labels = np.arange(topics_number)
    in_cluster = []
    out_cluster = []

    for idx in range(topics_number):
        cluster = df.loc[df[text] == selected_text, "label"].values[0]
        in_cluster_texts = df.loc[df[label] == cluster, text].values

        if len(in_cluster_texts) > 1:
            avg_in = 0
            for text_item in in_cluster_texts:
                avg_in += abs(df.loc[df[text] == text_item, str(idx)].values[0] - df.loc[df[text] == selected_text, str(idx)].values[0])
            avg_in /= (len(in_cluster_texts) - 1)
            in_cluster.append(avg_in)
        else:
            in_cluster.append(0)
        
        if (len(df[text].values) - len(in_cluster_texts)) > 0:
            avg_out = 0
            for text_item in df[text].values:
                if text_item in in_cluster_texts:
                    continue
                avg_out += abs(df.loc[df[text] == text_item, str(idx)].values[0] - df.loc[df[text] == selected_text, str(idx)].values[0])
            avg_out /= (len(df[text].values) - len(in_cluster_texts))
            out_cluster.append(avg_out)
        else:
            out_cluster.append(0)

    fig = go.Figure(data=[
        go.Bar(name='Inside cluster', x=np.arange(len(plot_labels)), y=in_cluster),
        go.Bar(name='Outside cluster', x=np.arange(len(plot_labels)), y=out_cluster)
    ])
    
    fig.update_xaxes(type='category')
    fig.update_layout(barmode='group')

    return fig


def plot_topic_distribution(df, selected_text, text="country", label = "label"):
    topics_number = len(df.columns) - 4
    fig, axes = plt.subplots(topics_number, 1, figsize=(15,1.5*topics_number))
    plt.subplots_adjust(hspace = 0.6)
    fig.suptitle(f"Topics distribution for {selected_text}")
    plt.xlim(0, 1)
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    fig.patch.set_facecolor('#262730')

    for ax in axes:
        ax.tick_params(color='white', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_facecolor('#262730')

    for idx in range(topics_number):
        axes[idx].set_title(f"Topic {idx}")
        axes[idx].axes.yaxis.set_ticklabels([])
        axes[idx].axvline(x = df.loc[df[text] == selected_text, str(idx)].values[0], color = 'g', lw = 3)
        axes[idx].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        cluster = df.loc[df[text] == selected_text, "label"].values[0]
        in_cluster_texts = df.loc[df[label] == cluster, text].values

        if len(in_cluster_texts) > 1:
            avg_in = df.loc[df[text].isin(in_cluster_texts), str(idx)].mean()
            axes[idx].axvline(x = avg_in, color = 'b', lw = 3)
        
        if (len(df[text].values) - len(in_cluster_texts)) > 0:
            avg_out = df.loc[~df[text].isin(in_cluster_texts), str(idx)].mean()
            axes[idx].axvline(x = avg_out, color = 'r', lw = 3)
        
    legend_lines = [plt.Line2D([0], [0], color="g", lw=3),
                    plt.Line2D([0], [0], color="b", lw=3),
                    plt.Line2D([0], [0], color="r", lw=3)]
    fig.legend(legend_lines, ['selected country', 'inside cluster', 'outside cluster'], facecolor = '#262730')

    return fig
