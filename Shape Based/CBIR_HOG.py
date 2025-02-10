import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def calculate_hog_features(image_path, resize_dim=(128, 128)): 
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

   
    image = cv2.resize(image, resize_dim)

    
    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True)

    return hog_features

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img_path))
    return images

def find_similar_images_hog(target_image_path, dataset_folder, num_similar=6):
    target_features = calculate_hog_features(target_image_path)
    images = load_images_from_folder(dataset_folder)

    distances = []
    for filename, img_path in images:
        features = calculate_hog_features(img_path)
        if features is not None:
            # Calculate Euclidean distance between HOG features
            distance = np.linalg.norm(target_features - features)
            distances.append((filename, img_path, distance))

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


target_image_path = '/content/823.jpg' 
dataset_folder = '/content'  


similar_images = find_similar_images_hog(target_image_path, dataset_folder, num_similar=6)


print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Distance: {distance}")

plot_images(similar_images, target_image_path)
