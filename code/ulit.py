import numpy as np

def Normalized(img):
    min_val = img.min()
    max_val = img.max()
    normalized_image = 255 * (img - min_val) / (max_val - min_val)
    normalized_image = normalized_image.astype(np.uint8)

    return normalized_image
