import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
  
def calculate_sift_features(image_path): 
     
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return descriptors

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img_path))
    return images

def find_similar_images_sift(target_image_path, dataset_folder, num_similar=6):
    target_descriptors = calculate_sift_features(target_image_path)
    images = load_images_from_folder(dataset_folder)

    bf = cv2.BFMatcher()
    distances = [] 
    for filename, img_path in images:
        descriptors = calculate_sift_features(img_path)
        if descriptors is not None:
            matches = bf.knnMatch(target_descriptors, descriptors, k=2)
            
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            distances.append((filename, img_path, len(good_matches)))

    distances.sort(key=lambda x: x[2], reverse=True)

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
        plt.title(f"Similar Image {i + 1}\nMatches: {distance}")
        plt.axis('off')

    plt.show()


target_image_path = '/content/322.jpg'
dataset_folder = '/content'
similar_images = find_similar_images_sift(target_image_path, dataset_folder, num_similar=6)

print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Matches: {distance}")

plot_images(similar_images, target_image_path)
