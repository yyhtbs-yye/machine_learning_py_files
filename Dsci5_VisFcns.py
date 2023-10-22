import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb
import numpy as np

# Function to plot decision boundaries for multiple classes
def scatter(X, y, xyeq=True):

    num_classes = len(np.unique(y))

    hues = np.linspace(0, 1, num_classes+1)[:-1]

    # Convert hues to RGB for light and bold versions
    cmap_bold = list([hsv_to_rgb((h, 1, 1)) for h in hues])

    plt.figure(figsize=(12, 3))

    for i, label in enumerate(np.unique(y)):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}", color=cmap_bold[label], edgecolor='k', s=20)

    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    if xyeq: plt.axis('scaled')
    plt.title("2D Data Visualisation")
    plt.show()
