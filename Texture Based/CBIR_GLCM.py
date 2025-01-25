import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def calculate_glcm_features(image_path, distances=[1], angles=[0], properties=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    glcm = greycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
 
    features = []
    for prop in properties:
        feature = greycoprops(glcm, prop).flatten()
        features.extend(feature)

    return features

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img_path))
    return images
 
def find_similar_images_glcm(target_image_path, dataset_folder, num_similar=5):
    target_features = calculate_glcm_features(target_image_path)
    images = load_images_from_folder(dataset_folder)

    distances = []
    for filename, img_path in images:
        features = calculate_glcm_features(img_path)
        dist = np.linalg.norm(np.array(target_features) - np.array(features))
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
similar_images = find_similar_images_glcm(target_image_path, dataset_folder, num_similar=6)

print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Distance: {distance}")

plot_images(similar_images, target_image_path)
