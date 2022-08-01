import plotly.express as px


def plot_clusters(df, x="c1", y="c2", color="label", text="country", marker_size=15):
    fig = px.scatter(df, x=x, y=y, color=color, text=text)
    fig.update_traces(marker={'size': marker_size})
    return fig
