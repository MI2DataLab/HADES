import json

class Config:
    def __init__(self, path):
        self.path = path
        self.load_settings_dict()
        self.load_default_config()

    def load_settings_dict(self):
        with open(self.path) as f:
            self.settings_dict = json.load(f)

    def load_default_config(self):
        self.mappings = ["tSNE", "UMAP"]
        self.clusterings = ["Hierarchical", "K-Means", "HDBSCAN"]
        self.metric_choices = {"ir": "information radius", "hd": "Hellinger distance"}
        self.default_config = {
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

    @property
    def sections(self):
        return list(self.settings_dict["sections"].keys())

    @property
    def id_column(self):
        return self.settings_dict["id_column"]
    
    @property
    def countries_division(self):
        return self.id_column.lower() == "country"



# ## Data settings
# COUNTRIES_DIVISION = False
# ID_COLUMN = None
# SETTINGS_DICT = None
# SECTIONS = None
# MAPPINGS = ["tSNE", "UMAP"]
# CLUSTERINGS = ["Hierarchical", "K-Means", "HDBSCAN"]
# METRIC_CHOICES = {"ir": "information radius", "hd": "Hellinger distance"}

# ## Streamlit settings
# DEFAULT_CONFIG = {
#     "displaylogo": False,
#     "staticPlot": False,
#     "toImageButtonOptions": {
#         "height": None,
#         "width": None,
#     },
#     "modeBarButtonsToRemove": [
#         "sendDataToCloud",
#         "lasso2d",
#         "autoScale2d",
#         "select2d",
#         "zoom2d",
#         "pan2d",
#         "zoomIn2d",
#         "zoomOut2d",
#         "resetScale2d",
#         "toggleSpikelines",
#         "hoverCompareCartesian",
#         "hoverClosestCartesian",
#     ],
# }

# def load_settings_dict(path):
#     global SETTINGS_DICT
#     global SECTIONS
#     global ID_COLUMN
#     global COUNTRIES_DIVISION
#     with open(path) as f:
#         SETTINGS_DICT = json.load(f)
#     SECTIONS = list(SETTINGS_DICT["sections"].keys())
#     ID_COLUMN = SETTINGS_DICT["id_column"]
#     COUNTRIES_DIVISION = ID_COLUMN.lower() == "country"
    
