import os
import random
import pandas as pd
import plotly.graph_objects as go
from matplotlib import cm
class Config:
    """
    Global configuration for data paths, topics, styling, and export settings.
    """
    INPUT_DIR = './bar_chart/csv'
    OUTPUT_PNG_DIR = './bar_chart/png'
    OUTPUT_SVG_DIR = './bar_chart/svg'

    # Predefined topics (must match CSV filenames)
    TOPICS = [
        "Transportation_and_Logistics", "Tourism_and_Hospitality", "Business_and_Finance",
        "Real_Estate_and_Housing_Market", "Healthcare_and_Health", "Retail_and_E-commerce",
        "Human_Resources_and_Employee_Management", "Sports_and_Entertainment", "Education_and_Academics",
        "Food_and_Beverage_Industry", "Science_and_Engineering", "Agriculture_and_Food_Production"
    ]

    # Available colormaps and hatch patterns
    RE_CMAPS = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "PuBu", "BuPu", "YlGn", "YlOrBr"]
    HATCHES = ["/", "\\", "-", "+", "x", "."]

    # Fill mode probabilities
    FILL_MODES = ["solid", "gradient", "pattern"]
    FILL_WEIGHTS = [0.7, 0.15, 0.15]

    # Export resolution settings
    BASE_DPI = 72      # Reference DPI for Plotly
    TARGET_DPI = 300   # Desired DPI for output images

    # Font and color settings (mirroring matplotlib rcParams)
    FONT_FAMILY = 'Times New Roman'
    FONT_COLOR = '#000000'
    AXIS_LINE_WIDTH = 1
    TICK_COLOR = '#000000'
    TICK_WIDTH = 1
    TICK_LEN = 5


class DataLoader:
    """
    Responsible for loading and preprocessing CSV data files.
    """
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def load(self, topic: str, index: int):
        """
        Load a CSV file for the given topic and index.
        Parses header for theme and unit, returns DataFrame and metadata.
        """
        infile = os.path.join(self.input_dir, f"bar_{topic}_{index}.csv")
        if not os.path.exists(infile):
            return None

        # Read header line for metadata
        with open(infile, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
        _, theme, unit, _ = [p.strip() for p in header_line.split(',')]

        # Read actual data: skip the first line, second line as header
        df = pd.read_csv(infile, skiprows=1)
        label_col = df.columns[0]
        df.columns = [label_col, unit]

        return df, label_col, unit, theme


class BarChartPlotter:
    """
    Builds and saves bar charts using Plotly, matching original matplotlib styles.
    """
    def __init__(self, config: Config):
        self.cfg = config
        # Tab20 colormap for solid fills
        self.tab20 = cm.get_cmap('tab20')

    def _rgba_to_plotly(self, rgba):
        """
        Convert a Matplotlib RGBA tuple (0-1 range) to Plotly 'rgb(r,g,b)' string.
        """
        r, g, b, _ = [int(255 * x) for x in rgba]
        return f'rgb({r},{g},{b})'

    def plot(self, df: pd.DataFrame, label_col: str, unit: str, theme: str,
             fill_mode: str, out_png: str, out_svg: str):
        labels = df[label_col].tolist()
        values = df[unit].to_numpy()
        n = len(values)

        # Determine orientation: more likely vertical ('v') or horizontal ('h')
        orientation = random.choices(['v', 'h'], weights=[0.7, 0.3])[0]

        # Configure marker based on fill_mode
        if fill_mode == 'solid':
            colors = [self._rgba_to_plotly(self.tab20(i % 20)) for i in range(n)]
            marker = dict(color=colors, line=dict(color=self.cfg.FONT_COLOR, width=self.cfg.AXIS_LINE_WIDTH))
        elif fill_mode == 'gradient':
            cmap_name = random.choice(self.cfg.RE_CMAPS)
            marker = dict(
                color=values,
                colorscale=cmap_name,
                line=dict(color=self.cfg.FONT_COLOR, width=self.cfg.AXIS_LINE_WIDTH),
                showscale=False
            )
        else:  # pattern
            hatch = random.choice(self.cfg.HATCHES)
            marker = dict(
                color='white',
                line=dict(color=self.cfg.FONT_COLOR, width=self.cfg.AXIS_LINE_WIDTH),
                pattern=dict(shape=hatch, fgcolor=self.cfg.FONT_COLOR)
            )

        # Build bar trace
        if orientation == 'v':
            trace = go.Bar(
                x=labels,
                y=values,
                marker=marker,
                text=values,
                textposition='outside',
                textfont=dict(size=(8 if n >= 10 else 11), family=self.cfg.FONT_FAMILY, color=self.cfg.FONT_COLOR)
            )
        else:
            trace = go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=marker,
                text=values,
                textposition='outside',
                textfont=dict(size=(8 if n >= 10 else 11), family=self.cfg.FONT_FAMILY, color=self.cfg.FONT_COLOR)
            )

        # Build figure and layout
        fig = go.Figure(trace)
        fig.update_layout(
            title=dict(text=f"{theme} ({unit})", font=dict(size=16, family=self.cfg.FONT_FAMILY, color=self.cfg.FONT_COLOR), x=0.5),
            font=dict(family=self.cfg.FONT_FAMILY, size=12, color=self.cfg.FONT_COLOR),
            plot_bgcolor='white',
            margin=dict(l=40, r=20, t=60, b=60)
        )
        # Axis styling with ticks
        axis_kwargs = dict(
            showline=True,
            linecolor=self.cfg.FONT_COLOR,
            linewidth=self.cfg.AXIS_LINE_WIDTH,
            ticks='outside',
            tickcolor=self.cfg.TICK_COLOR,
            tickwidth=self.cfg.TICK_WIDTH,
            ticklen=self.cfg.TICK_LEN,
            tickfont=dict(family=self.cfg.FONT_FAMILY, size=12, color=self.cfg.FONT_COLOR)
        )
        if orientation == 'v':
            fig.update_xaxes(title_text=label_col, tickangle=45, **axis_kwargs)
            fig.update_yaxes(title_text=unit, **axis_kwargs)
        else:
            fig.update_xaxes(title_text=unit, **axis_kwargs)
            fig.update_yaxes(title_text=label_col, **axis_kwargs)

        # Determine scale factor for desired DPI
        scale_factor = self.cfg.TARGET_DPI / self.cfg.BASE_DPI
        # Save outputs
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        os.makedirs(os.path.dirname(out_svg), exist_ok=True)
        fig.write_image(out_png, format='png', scale=scale_factor)
        fig.write_image(out_svg, format='svg')


if __name__ == '__main__':
    loader = DataLoader(Config.INPUT_DIR)
    plotter = BarChartPlotter(Config)
    count = 0

    # Iterate over topics and files
    # for topic in Config.TOPICS:
    for topic in ["Real_Estate_and_Housing_Market"]:
        for i in range(1, 9):
            result = loader.load(topic, i)
            if result is None:
                continue
            df, label_col, unit, theme = result
            fill_mode = random.choices(
                Config.FILL_MODES, weights=Config.FILL_WEIGHTS, k=1
            )[0]

            out_png = os.path.join(Config.OUTPUT_PNG_DIR, f"bar_{topic}_{i}.png")
            out_svg = os.path.join(Config.OUTPUT_SVG_DIR, f"bar_{topic}_{i}.svg")

            plotter.plot(df, label_col, unit, theme, fill_mode, out_png, out_svg)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} files...")
