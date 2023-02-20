import json

## Data settings
APP_SETTINGS_FILENAME = "app/app_settings_resilience_plans.json"
COUNTRIES_DIVISION = True
DIVISION_COLUMN = "country"
with open(APP_SETTINGS_FILENAME) as f:
    SETTINGS_DICT = json.load(f)
SECTIONS = list(SETTINGS_DICT["sections"].keys())
MAPPINGS = ["tSNE", "UMAP"]
CLUSTERINGS = ["Hierarchical", "K-Means", "HDBSCAN"]
METRIC_CHOICES = {"ir": "information radius", "hd": "Hellinger distance"}

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
