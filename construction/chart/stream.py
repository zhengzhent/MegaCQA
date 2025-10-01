import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.edgecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "figure.autolayout": False,
})

COLOR_CYCLE = plt.cm.tab20.colors
FIGSIZE = (8, 5)
DPI = 300

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def _parse_theme_and_units(first_line: str):
    """Return ``theme, field_name, unit`` parsed from the first metadata line."""
    parts = [p.strip() for p in first_line.split(',')]
    theme = parts[0] if parts else ""

    if len(parts) >= 3:  # Theme, Field, Unit
        return theme, parts[1], parts[2]

    field_unit_part = parts[1] if len(parts) > 1 else ""
    unit_match = re.search(r"\((.*?)\)", field_unit_part)
    if unit_match:
        unit = unit_match.group(1).strip()
        field_name = re.sub(r"\s*\(.*?\)", "", field_unit_part).strip()
    elif "." in field_unit_part:
        field_name, unit = [s.strip() for s in field_unit_part.split(".", 1)]
    else:
        field_name, unit = field_unit_part.strip(), ""

    return theme, field_name, unit


def _parse_headers(header_line: str):
    """Return column headers after Year."""
    return [col.strip() for col in header_line.split(',')[1:]]


def _smooth_series(x: np.ndarray, y: np.ndarray, factor: int = 10):
    """Interpolate one *y* series to make edges smooth.

    *factor* controls added resolution (10 → 10× more points).
    Uses *SciPy* cubic splines when available; falls back to linear ``numpy.interp``.
    """
    if len(x) < 4:
        return x, y  # nothing to smooth

    # New dense x‑grid
    x_new = np.linspace(x.min(), x.max(), len(x) * factor)

    try:
        from scipy.interpolate import make_interp_spline  # noqa: WPS433
        spline = make_interp_spline(x, y, k=3)
        y_new = spline(x_new)
    except Exception:  # SciPy missing or failed → linear fallback
        y_new = np.interp(x_new, x, y)

    # Replace any negative artefacts caused by smoothing
    y_new[y_new < 0] = 0
    return x_new, y_new


# -------------------------------------------------
# CORE PLOTTER
# -------------------------------------------------

def plot_stream_from_csv(csv_path: str, base_save_dir: str):
    """Read one CSV file and save a *streamgraph* (centered stacked area)."""
    try:
        # ---------------------------------  META
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()  # 0: Theme / Field / Unit
            header_line = f.readline().strip()  # 1: Year + columns
            f.readline()  # 2: Ignore trend / comment line

        theme, field_name, unit = _parse_theme_and_units(first_line)
        y_headers = _parse_headers(header_line)

        # ---------------------------------  DATA
        df = pd.read_csv(csv_path, skiprows=[0, 2], encoding="utf-8")
        years_raw = df.iloc[:, 0].astype(str).str.strip()
        years = pd.to_numeric(years_raw.str.extract(r"(\d+)")[0], errors="coerce")
        valid = ~years.isna()

        x = years[valid].astype(int).to_numpy()
        y_df = df.iloc[valid.index[valid], 1:].apply(pd.to_numeric, errors="coerce").fillna(0)

        # ---------------------------------  SMOOTHING
        # Each column independently smoothed on a common dense x grid
        x_smooth, first_series = _smooth_series(x, y_df.iloc[:, 0].to_numpy())
        y_smooth = np.empty((y_df.shape[1], len(x_smooth)))
        y_smooth[0, :] = first_series
        for idx in range(1, y_df.shape[1]):
            _, y_smooth[idx, :] = _smooth_series(x, y_df.iloc[:, idx].to_numpy())

        # ---------------------------------  PLOT
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        stack = ax.stackplot(
            x_smooth,
            y_smooth,
            colors=COLOR_CYCLE,
            labels=y_headers,
            baseline="sym",  # **key change** – center layers
            edgecolor="none",  # hide outlines for a cleaner stream look
        )

        # Titles & axis
        min_year, max_year = int(x.min()), int(x.max())
        title = f"{field_name} from {min_year} to {max_year} ({unit}) " if unit else f"{field_name} from {min_year} to {max_year} "

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel(df.columns[0], fontsize=12)
        # ax.set_ylabel(f"{field_name} ({unit})" if unit else field_name, fontsize=12) # 移除 Y 轴标签
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        step = max(1, len(x) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(x[::step].astype(int), rotation=45, ha="right")

        # Symmetric y‑limits 不用给顶上留空间
        total = y_smooth.sum(axis=0)
        effective_peak = total.max() / 2  # sym 基线下的可见半振幅
        y_max = effective_peak * 1.2  # 只留 120 % 的余量
        ax.set_ylim(-y_max, y_max)

        # 隐藏 Y 轴的刻度、刻度标签和轴线
        ax.yaxis.set_visible(False)


        # Legend on the right
        fig.subplots_adjust(right=0.8)
        ax.legend(
            handles=stack,
            labels=y_headers,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=10,
        )

        # ---------------------------------  SAVE
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        for ext in ("png", "svg"):
            out_dir = os.path.join(base_save_dir, ext)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight", dpi=DPI)
        plt.close()
        print(f"生成图表: {stem}")

    except Exception as exc:  # noqa: BLE001
        print(f"处理 {csv_path} 时出错: {exc}")
        if "fig" in locals():
            plt.close()


# -------------------------------------------------
# BATCH ENTRY‑POINT
# -------------------------------------------------

def batch_plot_stream(csv_dir="csv", output_dir="charts"):
    """Generate streamgraphs for all CSVs in *csv_dir*."""
    if not os.path.exists(csv_dir):
        print(f"目录不存在: {csv_dir}")
        return

    for csv in glob.glob(os.path.join(csv_dir, "*.csv")):
        print(f"正在处理: {os.path.basename(csv)}")
        plot_stream_from_csv(csv, output_dir)


if __name__ == "__main__":
    batch_plot_stream()
