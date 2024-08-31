import cv2
import numpy as np

def calculate_mean(image_path):
    # Read kar rahe hain image ko
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    mean_b = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_r = np.mean(image[:, :, 2])

    return (mean_b, mean_g, mean_r)

#Your image path
image_paths = ['/content/011.jpg'] #Your image path
#Your image path


for i, image_path in enumerate(image_paths):
    mean_values = calculate_mean(image_path)
    print(f"Image {i+1} ({image_path}) mean values: {mean_values}")
