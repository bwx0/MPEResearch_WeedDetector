import numpy as np
import matplotlib.pyplot as plt
import cv2


# Generate a sample grid with random data
def generate_sample_grid(grid_size):
    np.random.seed(50)
    return np.random.rand(grid_size, grid_size)


# Interpolate the grid to a higher resolution
def interpolate_grid(grid, method, upscale_factor):
    original_size = grid.shape
    new_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)

    # Map methods to OpenCV interpolation constants
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    if method not in interpolation_methods:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Resize using the specified method
    resized = cv2.resize(grid, new_size[::-1], interpolation=interpolation_methods[method])
    return resized


# Demonstrate various interpolation methods
def demonstrate_interpolation(grid_size=5, upscale_factor=10):
    methods = ["nearest", "bilinear", "bicubic", "area", "lanczos"]
    grid = generate_sample_grid(grid_size)

    # Set up the plot with a 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes = axes.ravel()  # Flatten the 3x2 array for easier indexing

    # Plot the original grid
    axes[0].imshow(grid, cmap="viridis", interpolation="none")
    axes[0].set_title("Original Grid")
    axes[0].axis("off")

    # Plot interpolations
    for i, method in enumerate(methods):
        interpolated_grid = interpolate_grid(grid, method, upscale_factor)
        axes[i + 1].imshow(interpolated_grid, cmap="viridis")
        axes[i + 1].set_title(method.capitalize())
        axes[i + 1].axis("off")

    # Remove unused subplot if there is any
    if len(methods) + 1 < len(axes):
        axes[len(methods) + 1].axis("off")

    plt.tight_layout()
    plt.show()


# Run the demonstration
demonstrate_interpolation(grid_size=8, upscale_factor=30)
