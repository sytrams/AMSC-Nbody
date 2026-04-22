# plot the dynamic file .git of the trajectories
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
OUTPUT_FILE = OUTPUT_DIR / "planets_animation.gif"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
BODY_MARKER_SIZE = 4


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


def build_planets(data):
    if data.shape[1] % 2 != 0:
        raise ValueError(f"Expected an even number of columns in {POSITIONS_FILE}, got {data.shape[1]}.")

    num_bodies = data.shape[1] // 2
    labels = build_body_labels(num_bodies)

    planets = []
    for index, label in enumerate(labels):
        planets.append(
            (
                label,
                2 * index,
                2 * index + 1,
                BODY_MARKER_SIZE,
                BODY_COLORS[index % len(BODY_COLORS)],
            )
        )
    return planets


# Load N-body positions
data = np.atleast_2d(np.loadtxt(POSITIONS_FILE, delimiter=","))
planets = build_planets(data)

# Setup figure
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Axis limits
min_x = data[:, 0::2].min()
max_x = data[:, 0::2].max()
min_y = data[:, 1::2].min()
max_y = data[:, 1::2].max()

# Add padding so planets aren't clipped
padding = max(max_x - min_x, max_y - min_y) * 0.03
if padding == 0:
    padding = 1.0

ax.set_xlim(min_x - padding, max_x + padding)
ax.set_ylim(min_y - padding, max_y + padding)

ax.set_aspect("equal")

# starry background
num_stars = 800
star_x = np.random.uniform(min_x - padding, max_x + padding, num_stars)
star_y = np.random.uniform(min_y - padding, max_y + padding, num_stars)
ax.scatter(star_x, star_y, s=1, color='white', alpha=0.2, zorder=0)

# Init planets
planet_points = []
planet_paths = []

for (name, xcol, ycol, radius, color) in planets:
    p = ax.plot([], [], 'o', markersize=radius, color=color)[0]
    t = ax.plot([], [], color=color, linewidth=1.2, alpha=0.7)[0]

    planet_points.append((p, xcol, ycol))
    planet_paths.append((t, xcol, ycol))

ax.set_xticks([])
ax.set_yticks([])

# Legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=name,
                              markerfacecolor=color, markersize=radius)
                   for (name, _, _, radius, color) in planets]

# put legend outside plot area
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), facecolor='black', framealpha=0.5, labelcolor='white')

# Animation
def animate(frame):
    draw_list = []

    for (p, xcol, ycol), (t, xcol2, ycol2) in zip(planet_points, planet_paths):

        # Update dot
        p.set_data([data[frame, xcol]], [data[frame, ycol]])
        draw_list.append(p)

        # Update trajectory
        t.set_data(data[:frame+1, xcol], data[:frame+1, ycol])
        draw_list.append(t)

    return draw_list

anim = FuncAnimation(fig, animate, frames=len(data), interval=20, blit=True)

print("Saving animation...")
anim.save(OUTPUT_FILE, writer='pillow', fps=30)
print("Saved!")
