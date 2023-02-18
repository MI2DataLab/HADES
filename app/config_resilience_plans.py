import json

## Data settings
APP_SETTINGS_FILENAME = "app_settings_resilience_plans.json"
COUNTRIES_DIVISION = True
DIVISION_COLUMN = "country"
with open(APP_SETTINGS_FILENAME) as f:
    SETTINGS_DICT = json.load(f)
SECTIONS = list(SETTINGS_DICT["sections"].keys())
MAPPINGS = ["tSNE", "UMAP"]
CLUSTERINGS = ["Hierarchical", "K-Means", "HDBSCAN"]
METRIC_CHOICES = {"ir": "information radius", "hd": "Hellinger distance"}

# Make LDAvis order
order_dict = {
    "Executive summary": [3, 2, 1],
    "Recovery and resilience challenges: scene-setter": [3, 1, 4, 2],
    "Objectives, structure and governance of the plan": [3, 1, 2, 4, 5, 6],
    "Summary of the assessment of the plan": [1, 6, 2, 3, 5, 4],
}

## Streamlit settings
DEFAULT_CONFIG = {
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
