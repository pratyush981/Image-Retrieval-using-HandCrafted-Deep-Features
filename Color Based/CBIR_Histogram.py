import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  
 
def calculate_histogram(image_path):
    image = cv2.imread(image_path) 
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.") 
 
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
 
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    
    return np.concatenate([hist_b, hist_g, hist_r])
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img_path))
    return images

def find_similar_images(target_image_path, dataset_folder, num_similar=10):
    target_hist = calculate_histogram(target_image_path)
    images = load_images_from_folder(dataset_folder)

    distances = []
    for filename, img_path in images:
        hist = calculate_histogram(img_path)
        distance = cv2.compareHist(target_hist, hist, cv2.HISTCMP_CHISQR)
        distances.append((filename, img_path, distance))
 
    distances.sort(key=lambda x: x[2])
    similar_images = distances[:num_similar]

    return similar_images

def plot_images(similar_images, target_image_path):
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(similar_images) + 1, 1)
    plt.imshow(target_image)
    plt.title("Target Image")
    plt.axis('off')

    for i, (filename, img_path, distance) in enumerate(similar_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(similar_images) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Similar Image {i + 1}\nDistance: {distance:.2f}")
        plt.axis('off')

    plt.show()
#Your path HERE
target_image_path = '/content/330.jpg'
dataset_folder = '/content'

similar_images = find_similar_images(target_image_path, dataset_folder)

print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Distance: {distance}")

plot_images(similar_images, target_image_path)
