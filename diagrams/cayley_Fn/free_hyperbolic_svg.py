from orthogonal_curve import calculate_orthogonal_circle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Warm-leaning neutral palette: near-black + gray with subtle warm tint
color_a = '#141210'  # near-black, slightly warm (generator a)
color_b = '#5e5a55'  # neutral gray, slightly warm (generator b)

LW_AXIS = 1.2
LW_ARC = 0.9


def _get_pair(base, diff):
    A = np.array([np.cos(base - diff), np.sin(base - diff)])
    B = np.array([np.cos(base + diff), np.sin(base + diff)])
    return A, B


def _draw_orthogonal_arc(ax, A, B, arc_color):
    center, radius = calculate_orthogonal_circle(A, B)
    d = np.linalg.norm(center)
    a = (1 + d ** 2 - radius ** 2) / (2 * d)
    h = np.sqrt(max(0, 1 - a ** 2))
    x1 = a * center[0] / d + h * center[1] / d
    y1 = a * center[1] / d - h * center[0] / d
    x2 = a * center[0] / d - h * center[1] / d
    y2 = a * center[1] / d + h * center[0] / d

    start_angle = np.arctan2(y1 - center[1], x1 - center[0])
    end_angle = np.arctan2(y2 - center[1], x2 - center[0])
    if np.cross(center, [x1, y1]) < 0:
        start_angle, end_angle = end_angle, start_angle

    arc = Arc(center, 2 * radius, 2 * radius, angle=0,
              theta1=np.degrees(start_angle), theta2=np.degrees(end_angle),
              color=arc_color, linewidth=LW_ARC)
    ax.add_artist(arc)


def _draw(ax, m):
    base = np.pi / int(m)
    diff = np.pi / int(3 * m)
    for n in range(2 * m):
        A, B = _get_pair(n * base, diff)
        color = color_a if n % 2 == 0 else color_b
        _draw_orthogonal_arc(ax, A, B, color)


fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Two generator axes: horizontal = b (lighter), vertical = a (darker)
ax.plot([-1, 1], [0, 0], color=color_b, linewidth=LW_AXIS)
ax.plot([0, 0], [-1, 1], color=color_a, linewidth=LW_AXIS)

depth = 7
m = 2
for _ in range(depth):
    _draw(ax, m)
    m *= 3

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')

plt.savefig("free_hyperbolic.svg", bbox_inches='tight', pad_inches=0, transparent=True)
