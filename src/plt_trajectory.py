# plot the static trajectory of the position in time of the bodies
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
BUILD_DIR = PROJECT_DIR / "build"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
BUILD_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

positions_candidates = [
    BUILD_DIR / "positions.txt",
    BUILD_DIR / "debug" / "positions.txt",
    PROJECT_DIR / "positions.txt",
]
existing_positions = [path for path in positions_candidates if path.exists()]
if not existing_positions:
    raise FileNotFoundError("No positions.txt found in build/, build/debug/, or the project root.")

POSITIONS_FILE = max(existing_positions, key=lambda path: path.stat().st_mtime)
OUTPUT_DIR = POSITIONS_FILE.parent if POSITIONS_FILE.parent != PROJECT_DIR else BUILD_DIR
OUTPUT_FILE = OUTPUT_DIR / "trajectory.png"

import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non interattivo
import matplotlib.pyplot as plt


BODY_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def build_body_labels(num_bodies):
    solar_system_file = PROJECT_DIR / "src" / "solar_system.txt"
    labels = []
    if solar_system_file.exists():
        for line in solar_system_file.read_text(encoding="utf-8").splitlines()[1:]:
            stripped = line.strip()
            if stripped:
                labels.append(stripped.split()[0])

    body_labels = []
    for index in range(num_bodies):
        if index < len(labels):
            body_labels.append(labels[index])
        else:
            body_labels.append(f"Body {index + 1}")
    return body_labels


def build_body_specs(data):
    if data.shape[1] % 2 != 0:
        raise ValueError(f"Expected an even number of columns in {POSITIONS_FILE}, got {data.shape[1]}.")

    num_bodies = data.shape[1] // 2
    labels = build_body_labels(num_bodies)

    body_specs = []
    for index, label in enumerate(labels):
        body_specs.append(
            {
                "label": label,
                "xcol": 2 * index,
                "ycol": 2 * index + 1,
                "color": BODY_COLORS[index % len(BODY_COLORS)],
            }
        )
    return body_specs


# Load the data from the file
data = np.atleast_2d(np.loadtxt(POSITIONS_FILE, delimiter=","))
body_specs = build_body_specs(data)

# Create a new figure
plt.figure()

# Plot the data
for body in body_specs:
    plt.plot(
        data[:, body["xcol"]],
        data[:, body["ycol"]],
        linewidth=1,
        color=body["color"],
        label=body["label"],
    )

# Add a legend
plt.legend()
x_values = data[:, 0::2]
y_values = data[:, 1::2]
min_x, max_x = x_values.min(), x_values.max()
min_y, max_y = y_values.min(), y_values.max()
padding = max(max_x - min_x, max_y - min_y) * 0.05
if padding == 0:
    padding = 1.0

plt.xlim(min_x - padding, max_x + padding)
plt.ylim(min_y - padding, max_y + padding)
plt.gca().set_aspect("equal", adjustable="box")


# Show the plot      plt.show()
plt.savefig(OUTPUT_FILE)  # salva l’immagine

print(f"Plot salvato in {OUTPUT_FILE}")
