import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity
    distributions of two images by computing the intersection of their
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values
    in each corresponding bin of the two normalized histograms. The result
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    # Step 1: Compute histograms for both images
    img1_hist = np.histogram(img1, bins=256, range=(0, 256))[0]
    img1_hist = img1_hist.astype(np.float32)
    img2_hist = np.histogram(img2, bins=256, range=(0, 256))[0]
    img2_hist = img2_hist.astype(np.float32)

    # Step 2: Normalize histograms
    img1_hist = img1_hist / np.sum(img1_hist)
    img2_hist = img2_hist / np.sum(img2_hist)

    # Step 3: Compute histogram intersection
    intersection = np.sum(np.minimum(img1_hist, img2_hist))

    ### END CODE HERE ###


    return float(intersection)


def plot_histograms_and_intersection(img1, img2, img1_hist, img2_hist):
    """
    Plot the histograms of two images and their intersection.
    """
    plt.figure(figsize=(15, 5))

    # Plot first image
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')

    # Plot second image
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    # Plot histograms and intersection
    plt.subplot(1, 3, 3)
    bins = np.arange(256)

    # Plot individual histograms
    plt.plot(bins, img1_hist, 'b-', alpha=0.7, label='Image 1', linewidth=2)
    plt.plot(bins, img2_hist, 'r-', alpha=0.7, label='Image 2', linewidth=2)

    # Plot intersection (minimum of both histograms)
    intersection_hist = np.minimum(img1_hist, img2_hist)
    plt.fill_between(bins, intersection_hist, alpha=0.5,
                        color='green', label='Intersection')

    plt.xlabel('Intensity Value')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram Intersection')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img1_path = r"C:\Users\manoe\OneDrive\Desktop\Programming\dip-2025-1\img\SanFrancisco.jpg"
    img2_path = r"C:\Users\manoe\OneDrive\Desktop\Programming\dip-2025-1\img\redacao.png"

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Compute histogram intersection score
    score = compute_histogram_intersection(img1, img2)
    print(f"Histogram Intersection Score: {score:.4f}")

    # Compute normalized histograms for plotting
    img1_hist = np.histogram(img1, bins=256, range=(0, 256))[0].astype(np.float32)
    img2_hist = np.histogram(img2, bins=256, range=(0, 256))[0].astype(np.float32)
    img1_hist = img1_hist / np.sum(img1_hist)
    img2_hist = img2_hist / np.sum(img2_hist)

    # Plot the results
    plot_histograms_and_intersection(img1, img2, img1_hist, img2_hist)
