import numpy as np
import matplotlib.pyplot as plt

def drawPolyakovLoop():

   

    # Define grid dimensions and loop parameters
    grid_size = 6
    loop_line_width = 3  # Increased line width for loops
    grid_line_width = 0.5  # Increased line width for the grid

    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw the lattice lines with increased thickness
    for x in range(grid_size + 1):
        ax.axhline(y=x, color='black', linewidth=grid_line_width)
        ax.axvline(x=x, color='black', linewidth=grid_line_width)

    # Draw Polyakov loop with an arrow
    ax.arrow(6, -1, 0, 7, head_width=0.2, head_length=0.2, fc='blue', ec='blue', linewidth=loop_line_width, length_includes_head=True)
    ax.text(4.2, -0.5, 'Polyakov loop', color='blue', fontsize=8)

    # Draw trivial loops with arrows and increased line width
    trivial_loops = [
        [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)],
        [(3, 1), (3, 2), (4, 2), (4, 3), (3, 3),(3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (5, 1), (4, 1), (3, 1)]
    ]

    for loop in trivial_loops:
        for i in range(len(loop) - 1):
            start = loop[i]
            end = loop[i + 1]
            # Draw the arrow for each segment of the loop
            ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                    head_width=0.1, head_length=0.1, fc='red', ec='red',
                    linewidth=loop_line_width, length_includes_head=True)

    ax.text(2.5, 0.5, 'Trivial loops', color='red', fontsize=8, ha='center')

    # Draw boundary condition line with increased thickness
    ax.plot([0, grid_size], [grid_size, grid_size], color='magenta', linestyle='--', linewidth=2)
    ax.text(3, 7, 'Boundary condition', color='magenta', fontsize=8, ha='center')

    # Mark the Z's
    ax.text(0.5, grid_size + 0.3, 'g', color='magenta', fontsize=8, ha='center')
    ax.text(5.5, grid_size + 0.3, 'g', color='magenta', fontsize=8, ha='center')
    ax.text(2.5, grid_size + 0.3, 'g', color='magenta', fontsize=8, ha='center')

    # Set the limits and aspect ratio
    ax.set_xlim(-1, grid_size + 1)
    ax.set_ylim(-1, grid_size + 1)
    ax.set_aspect('equal')

    # Turn off the axis
    plt.axis('off')

    # Show the plot
    plt.show()


if __name__ =='__main__':
    drawPolyakovLoop()