import os
import random
import pandas as pd
import matplotlib.pyplot as plt


class BoxPlotGenerator:
    """
    Generates box plot charts from CSV files, maintaining consistent style parameters.
    """

    BOX_COLORS = [
        '#FFFFFF',  # White
        '#F0F0F0',  # Light gray
        '#E6E6FA',  # Lavender
        '#F5F5DC',  # Beige
        '#F0FFF0',  # Honeydew
        '#FFF0F5',  # Lavender blush
        '#F0F8FF',  # Alice blue
        '#FAF0E6',  # Linen
        '#F5F5F5',  # White smoke
        '#E0FFFF',  # Light cyan
        '#D3D3D3',  # Light gray (darker)
        '#B0C4DE',  # Light steel blue
        '#DDA0DD',  # Plum
        '#98FB98',  # Pale green
        '#FFB6C1',  # Light pink
        '#87CEEB',  # Sky blue
        '#D8BFD8',  # Thistle
        '#F0E68C',  # Khaki
        '#E6E6FA',  # Lavender
        '#FFE4E1',  # Misty rose
        '#B0E0E6',  # Powder blue
        '#DCDCDC',  # Gainsboro
        '#F5DEB3',  # Wheat
        '#E0FFFF',  # Light cyan
        '#F5F5F5',  # White smoke
        '#778899',  # Light slate gray
        '#B8860B',  # Dark goldenrod
        '#4682B4',  # Steel blue
        '#556B2F',  # Dark olive green
        '#8B4513',  # Saddle brown
        '#483D8B',  # Dark slate blue
        '#2F4F4F',  # Dark slate gray
        '#8B008B',  # Dark magenta
        '#800000',  # Maroon
        '#4B0082',  # Indigo
    ]

    TOPICS = [
        "Transportation_and_Logistics", "Tourism_and_Hospitality", "Business_and_Finance",
        "Real_Estate_and_Housing_Market", "Healthcare_and_Health", "Retail_and_E-commerce",
        "Human_Resources_and_Employee_Management", "Sports_and_Entertainment", "Education_and_Academics",
        "Food_and_Beverage_Industry", "Science_and_Engineering", "Agriculture_and_Food_Production",
        "Energy_and_Utilities", "Cultural_Trends_and_Influences", "Social_Media_and_Digital_Media_and_Streaming"
    ]

    def __init__(
        self,
        # input_dir: str = './box_plot/csv',
        # output_png_dir: str = './box_plot/png',
        # output_svg_dir: str = './box_plot/svg',
        input_dir: str = './csv/box_plot',
        output_png_dir: str = './png/box_plot',
        output_svg_dir: str = './svg/box_plot',
        topic_indices: list = None,
        max_files_per_topic: int = 800
    ):
        self.input_dir = input_dir
        self.output_png_dir = output_png_dir
        self.output_svg_dir = output_svg_dir
        self.topic_indices = topic_indices or [1]
        self.max_files = max_files_per_topic
        self._ensure_output_dirs()
        self._configure_styles()

    def _ensure_output_dirs(self):
        os.makedirs(self.output_png_dir, exist_ok=True)
        os.makedirs(self.output_svg_dir, exist_ok=True)

    def _configure_styles(self):
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'text.color': '#000000',
            'axes.labelcolor': '#000000',
            'axes.edgecolor': '#000000',
            'xtick.color': '#000000',
            'ytick.color': '#000000',
            'axes.titleweight': 'bold',
            'axes.titlepad': 12,
            'grid.color': 'none',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.linewidth': 1.0,
        })

    def _read_metadata(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            columns = f.readline().strip()
        topic_name, theme, unit, dist_type, orientation = [p.strip() for p in header.split(',')]
        labels = columns.split(',')
        return theme, unit, labels, orientation

    def _load_data(self, filepath: str, labels: list):
        df = pd.read_csv(filepath, skiprows=2, header=None)
        df.columns = labels
        return df

    def _plot_box(self, df: pd.DataFrame, theme: str, unit: str, labels: list, orientation:bool):

        data = [df[label].values for label in labels]
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
        ax.set_title(f'{theme} ({unit})', fontsize=16)

        width = min(0.6, max(0.2, 0.8 / len(labels)))
        flierprops = {
            'marker': 'o',
            'markersize': random.uniform(2, 4),
            'markerfacecolor': '#000000',
            'alpha': 1.0
        }
        boxprops = {'linewidth': 1.5, 'edgecolor': '#000000'}
        whiskerprops = {'linewidth': 1.5, 'color': '#000000'}
        capprops = {'linewidth': 1.5, 'color': '#000000'}
        medianprops = {'linewidth': 1.5, 'color': '#000000'}

        box = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            widths=width,
            # whis=1.0,
            flierprops=flierprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            vert=orientation
        )

        # Color each box uniquely
        colors = random.sample(self.BOX_COLORS, len(box['boxes']))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Axis label and tick adjustments based on orientation
        if orientation:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlabel('', fontsize=14)
            ax.set_ylabel(unit, fontsize=14)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        else:
            ax.set_yticklabels(labels, rotation=0, va='center')
            ax.set_xlabel(unit, fontsize=14)
            ax.set_ylabel('', fontsize=14)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

        return fig

    def run(self):
        j = 1
        for t in self.topic_indices:
            for i in range(1, self.max_files + 1):
        # for t in range(12,13):
        #     for i in range(1, 20):
                filename = f'box_plot_{self.TOPICS[t-1]}_{i}.csv'
                input_path = os.path.join(self.input_dir, filename)
                if not os.path.exists(input_path):
                    continue

                theme, unit, labels, orientation = self._read_metadata(input_path)
                df = self._load_data(input_path, labels)
                fig = self._plot_box(df, theme, unit, labels, orientation= (orientation == 'vertical') )

                out_png_path = os.path.join(self.output_png_dir, f'box_plot_{self.TOPICS[t-1]}_{i}.png')
                out_svg_path = os.path.join(self.output_svg_dir, f'box_plot_{self.TOPICS[t-1]}_{i}.svg')
                fig.savefig(out_png_path, format='png', dpi=300, bbox_inches='tight')
                fig.savefig(out_svg_path, format='svg', dpi=300, bbox_inches='tight')
                plt.close(fig)

                if j % 100 == 0:
                    print(f'Processed {j} files')
                j += 1

        print(f'Completed processing, total files: {j - 1}')


if __name__ == '__main__':
    generator = BoxPlotGenerator(topic_indices=range(1,16), max_files_per_topic=1000)
    generator.run()
