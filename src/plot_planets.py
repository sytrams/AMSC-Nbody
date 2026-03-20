# plot the dynamic file .git of the trajectories
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load N-body positions
data = np.loadtxt("../build/positions.txt", delimiter=",")

# Planets (name, xcol, ycol, radius, color)
planets = [
    ('Sun',     0, 1,    9, 'yellow'),
    ('Mercury', 2, 3,    2, 'gray'),
    ('Venus',   4, 5,    4, 'orange'),
    ('Earth',   6, 7,    4, 'dodgerblue'),
    ('Mars',    8, 9,    3, 'red'),
    ('Jupiter', 10,11,   7, 'sandybrown'),
    ('Uranus',  12,13,   5, 'cyan'),
    ('Neptune', 14,15,   5, 'deepskyblue')
]

# Setup figure
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Axis limits
cols_x = [xcol for (_, xcol, _, _, _) in planets]
cols_y = [ycol for (_, _, ycol, _, _) in planets]

min_x = data[:, cols_x].min()
max_x = data[:, cols_x].max()
min_y = data[:, cols_y].min()
max_y = data[:, cols_y].max()

# Add padding so planets aren't clipped
padding = (max_x - min_x) * 0.03   # 3% of width

ax.set_xlim(min_x - padding, max_x + padding)
ax.set_ylim(min_y - padding, max_y + padding)

ax.set_aspect("auto")   # nothing gets stretched
#ax.set_aspect("equal")

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
anim.save("../build/planets_animation.gif", writer='pillow', fps=30)
print("Saved!")
