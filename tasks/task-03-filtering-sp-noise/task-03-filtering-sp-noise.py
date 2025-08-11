import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    # TODO: Implement noise removal here (e.g., median filtering)

    #as salt and pepper is a noise based on dark and brigth pixels, we can use a median filter.

    image = cv2.medianBlur(image, 5)

    return image

if __name__ == "__main__":
    noisy_image = cv2.imread(r"C:\Users\manoe\OneDrive\Desktop\Programming\dip-2025-1\img\head.png", cv2.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv2.imwrite("denoised_image.png", denoised_image)
