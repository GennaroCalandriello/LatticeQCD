import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define the parameters for the diagram
n_paths = 5  # Number of paths to draw
n_points = 20  # Number of points per path
amplitude = 3  # Amplitude of the wave
x = np.linspace(0, 10, n_points)  # x-axis
y_classical = amplitude * np.sin(np.pi * x / max(x))  # Classical path

# Plot the classical path
plt.plot(x, y_classical, label="classical solution", color="red")

# Create a function to generate random paths around the classical path
def generate_paths(x, y_classical, n_paths, spread=1.0):
    paths = []
    for i in range(n_paths):
        random_offsets = spread * (np.random.rand(len(x)) - 0.5)
        path = y_classical + random_offsets
        paths.append(path)
    return paths

# Generate random paths
paths = generate_paths(x, y_classical, n_paths)

# Plot the random paths
for path in paths:
    plt.plot(x, path, color="blue", alpha=0.5)

# Highlight the area between two selected paths to represent a "bundle" of paths
bundle = Polygon(np.column_stack([x, paths[1]]), closed=False, facecolor="blue", edgecolor="blue", alpha=0.1)
plt.gca().add_patch(bundle)
plt.fill_between(x, paths[1], paths[2], color="blue", alpha=0.1)

# Set the labels and title
plt.xlabel('x')
plt.ylabel('ϕ(x)')
plt.title('Path Integral Visualization')

# Remove y-axis ticks
plt.yticks([])

# Draw vertical lines to represent discrete points in the path integral
for xi in x:
    plt.axvline(xi, color='grey', linestyle='--', alpha=0.4)

# Add legend
plt.legend()

# Show the plot
plt.show()
