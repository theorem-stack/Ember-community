import numpy as np
import matplotlib.pyplot as plt

def plot_mask(mask: np.ndarray, title: str = "Mask Visualization", cmap: str = "viridis") -> None:
    """
    Plot a 2D mask using Matplotlib as a heatmap.

    Args:
        mask (np.ndarray): 2D NumPy array (boolean or numeric) to visualize.
        title (str): Title of the plot (default: "Mask Visualization").
        cmap (str): Colormap for the heatmap (default: "viridis").
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Mask Value')
    plt.title(title)
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.grid(False)
    plt.show()