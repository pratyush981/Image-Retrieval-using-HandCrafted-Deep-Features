import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.spatial import distance

def calculate_lbp(image_path, radius=1, n_points=8):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img_path))
    return images

def find_similar_images_lbp(target_image_path, dataset_folder, num_similar=5):
    target_hist = calculate_lbp(target_image_path)
    images = load_images_from_folder(dataset_folder)

    distances = []
    for filename, img_path in images:
        hist = calculate_lbp(img_path)
        dist = distance.euclidean(target_hist, hist)
        distances.append((filename, img_path, dist))

    distances.sort(key=lambda x: x[2])
    similar_images = distances[:num_similar]

    return similar_images

def plot_images(images, target_image_path):
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(images) + 1, 1)
    plt.imshow(target_image)
    plt.title("Target Image")
    plt.axis('off')

    for i, (filename, img_path, distance) in enumerate(images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(images) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Similar Image {i + 1}\nDistance: {distance:.2f}")
        plt.axis('off')

    plt.show()

target_image_path = '/content/322.jpg'
dataset_folder = '/content'

# Find the top 5 similar images based on LBP
similar_images = find_similar_images_lbp(target_image_path, dataset_folder, num_similar=6)

print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Distance: {distance}")

plot_images(similar_images, target_image_path)
