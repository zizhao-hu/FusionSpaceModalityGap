"""
Material Design colors for consistent use throughout the project.
This file contains a comprehensive set of colors for visualizations.
"""

# Primary colors (lighter versions)
PRIMARY = "#42A5F5"  # Blue 400 (lighter)
PRIMARY_LIGHT = "#90CAF9"  # Blue 200 (even lighter)
PRIMARY_DARK = "#1976D2"  # Blue 700

# Secondary colors (lighter versions)
SECONDARY = "#FFB74D"  # Orange 300 (lighter)
SECONDARY_LIGHT = "#FFCC80"  # Orange 200 (even lighter)
SECONDARY_DARK = "#F57C00"  # Orange 700

# Tertiary colors
TERTIARY = "#66BB6A"  # Green 400
TERTIARY_LIGHT = "#A5D6A7"  # Green 200
TERTIARY_DARK = "#388E3C"  # Green 700

# Accent colors (lighter versions)
ACCENT = "#B388FF"  # Deep Purple A100 (lighter)
ACCENT_LIGHT = "#D1C4E9"  # Deep Purple 100 (even lighter)
ACCENT_DARK = "#7C4DFF"  # Deep Purple A200

# Semantic colors (lighter versions)
SUCCESS = "#81C784"  # Green 300 (lighter)
WARNING = "#FFD54F"  # Amber 300 (lighter)
ERROR = "#EF5350"  # Red 400 (lighter)
INFO = "#64B5F6"  # Blue 300 (lighter)

# Neutral colors
GREY_50 = "#FAFAFA"
GREY_100 = "#F5F5F5"
GREY_200 = "#EEEEEE"
GREY_300 = "#E0E0E0"
GREY_400 = "#BDBDBD"
GREY_500 = "#9E9E9E"
GREY_600 = "#757575"
GREY_700 = "#616161"
GREY_800 = "#424242"
GREY_900 = "#212121"

# Chart colors (lighter palette)
CHART_COLORS = [
    "#64B5F6",  # Blue 300
    "#EF5350",  # Red 400
    "#81C784",  # Green 300
    "#FFD54F",  # Amber 300
    "#BA68C8",  # Purple 300
    "#FFB74D",  # Orange 300
    "#A1887F",  # Brown 300
    "#90A4AE",  # Blue Grey 300
]

# Gender visualization colors (lighter)
GEN_0_COLOR = GREY_400  # Lighter grey
GEN_10_INCREASE_COLOR = "#EF5350"  # Red 400 (lighter)
GEN_10_DECREASE_COLOR = "#64B5F6"  # Blue 300 (lighter)
GEN_10_NEUTRAL_COLOR = "#64B5F6"  # Blue 300 (lighter)

# Ethnicity visualization colors (lighter)
ETHNICITY_COLORS = {
    "white": "#90CAF9",  # Blue 200
    "black": "#CE93D8",  # Purple 200
    "asian": "#80CBC4",  # Teal 200
    "indian": "#FFAB91",  # Deep Orange 200
    "middle eastern": "#FFCC80",  # Orange 200
    "latino hispanic": "#A5D6A7",  # Green 200
}

# Additional color palettes

# Pastel palette
PASTEL_COLORS = [
    "#BBDEFB",  # Blue 100
    "#FFCDD2",  # Red 100
    "#C8E6C9",  # Green 100
    "#FFF9C4",  # Yellow 100
    "#E1BEE7",  # Purple 100
    "#FFE0B2",  # Orange 100
    "#D7CCC8",  # Brown 100
    "#CFD8DC",  # Blue Grey 100
]

# Vibrant palette
VIBRANT_COLORS = [
    "#2979FF",  # Blue A400
    "#FF1744",  # Red A400
    "#00E676",  # Green A400
    "#FFEA00",  # Yellow A400
    "#D500F9",  # Purple A400
    "#FF9100",  # Orange A400
    "#FF3D00",  # Deep Orange A400
    "#00B0FF",  # Light Blue A400
]

# Gradient palettes
BLUE_GRADIENT = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3", "#1E88E5", "#1976D2", "#1565C0", "#0D47A1"]
RED_GRADIENT = ["#FFEBEE", "#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#F44336", "#E53935", "#D32F2F", "#C62828", "#B71C1C"]
GREEN_GRADIENT = ["#E8F5E9", "#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A", "#4CAF50", "#43A047", "#388E3C", "#2E7D32", "#1B5E20"]
PURPLE_GRADIENT = ["#F3E5F5", "#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC", "#9C27B0", "#8E24AA", "#7B1FA2", "#6A1B9A", "#4A148C"]
ORANGE_GRADIENT = ["#FFF3E0", "#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726", "#FF9800", "#FB8C00", "#F57C00", "#EF6C00", "#E65100"]

# Diverging palettes (for showing positive/negative values)
RED_BLUE_DIVERGING = ["#B71C1C", "#D32F2F", "#EF5350", "#E57373", "#EF9A9A", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#1976D2"]
PURPLE_GREEN_DIVERGING = ["#4A148C", "#7B1FA2", "#AB47BC", "#CE93D8", "#E1BEE7", "#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A", "#2E7D32"]
ORANGE_BLUE_DIVERGING = ["#E65100", "#F57C00", "#FFA726", "#FFB74D", "#FFCC80", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#1976D2"]

# Categorical palette (for distinct categories)
CATEGORICAL_COLORS = [
    "#42A5F5",  # Blue 400
    "#EF5350",  # Red 400
    "#66BB6A",  # Green 400
    "#FFCA28",  # Amber 400
    "#AB47BC",  # Purple 400
    "#FFA726",  # Orange 400
    "#8D6E63",  # Brown 400
    "#26A69A",  # Teal 400
    "#EC407A",  # Pink 400
    "#5C6BC0",  # Indigo 400
]

# Default color for unspecified categories
DEFAULT_COLOR = GREY_400 