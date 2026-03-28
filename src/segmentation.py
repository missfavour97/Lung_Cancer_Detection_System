import cv2
import numpy as np

def segment_lung(image):
    # Convert PIL image to numpy array
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to separate lung region from background
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert so lungs become white in the mask
    thresh = cv2.bitwise_not(thresh)

    # Morphological cleaning
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Light smoothing to reduce small internal dots
    cleaned = cv2.medianBlur(cleaned, 7)

    # Apply lung mask to original image
    masked_image = cv2.bitwise_and(img, img, mask=cleaned)

    return masked_image