X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

SEPARATOR_TOKENS = [
    X_START,
    X_END,
    Y_START,
    Y_END,
]

LINE_TOKEN =  "<line>" 
VERTICAL_BAR_TOKEN = "<vertical_bar>"
HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
DOT_TOKEN = "<dot>"
HISTOGRAM_TOKEN = "<histogram>"

CHART_TYPE_TOKENS = [
    LINE_TOKEN,
    VERTICAL_BAR_TOKEN,
    HORIZONTAL_BAR_TOKEN,
    DOT_TOKEN,
    HISTOGRAM_TOKEN,
]

NEW_TOKENS = SEPARATOR_TOKENS + CHART_TYPE_TOKENS