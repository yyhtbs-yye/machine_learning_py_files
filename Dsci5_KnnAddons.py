from matplotlib.colors import ListedColormap, hsv_to_rgb
import numpy as np

# Function to plot decision boundaries for multiple classes
def knn_visualisation(X_train, y_train, X_test, y_test, knn_fcn, ax, k=3, reslx=20):

    num_classes = len(np.unique(y_train))

    hues = np.linspace(0, 1, num_classes+1)[:-1]

    # Convert hues to RGB for light and bold versions
    cmap_light = ListedColormap([hsv_to_rgb((h, 0.5, 1)) for h in hues])
    cmap_bold = list([hsv_to_rgb((h, 1, 1)) for h in hues])
    
    # Find the min and max values for each feature
    x0_min, x0_max = X_train[:, 0].min(), X_train[:, 0].max()
    x1_min, x1_max = X_train[:, 1].min(), X_train[:, 1].max()
    
    # Generate a mesh grid
    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, reslx), np.linspace(x1_min, x1_max, reslx))
    
    # Make predictions on the mesh grid points
    Z = knn_fcn(X_train, y_train, np.c_[xx0.ravel(), xx1.ravel()], k=k)
    Z = Z.reshape(xx0.shape)
    
    # Plot the decision boundary
    ax.contourf(xx0, xx1, Z, alpha=0.8, cmap=cmap_light)
    
    # Loop through each unique class label and plot the points
    for i, label in enumerate(np.unique(y_train)):
        ax.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], label=f"Train Class {label}", color=cmap_bold[label], edgecolor='k', s=20)
        ax.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1], marker='^', label=f"Pred. Class {label}", color=cmap_bold[label], edgecolor='k', s=80)
    
    # Customize the plot
    ax.set_xlim((x0_min, x0_max))
    ax.set_ylim((x1_min, x1_max))
    ax.set_aspect('equal', adjustable='box')

    ax.legend()
