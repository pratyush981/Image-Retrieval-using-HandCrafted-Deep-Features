import cv2 
import numpy as np 
import os  
import matplotlib.pyplot as plt 
  
def calculate_mean(image_path):
    # Read the image  
    image = cv2.imread(image_path)  
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    mean_b = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_r = np.mean(image[:, :, 2])
 
    return (mean_b, mean_g, mean_r)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((filename, img_path))
    return images

def find_similar_images(target_image_path, dataset_folder, num_similar=6):
    target_mean = calculate_mean(target_image_path)
    images = load_images_from_folder(dataset_folder)

    distances = []
    for filename, img_path in images:
        mean = calculate_mean(img_path)
        # Euclidean
        distance = np.linalg.norm(np.array(target_mean) - np.array(mean))
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
        plt.title(f"Similar Image {i+1}\nDistance: {distance:.2f}")
        plt.axis('off')

    plt.show()
#your path should appear in the output
target_image_path = '/content/006.jpg'
dataset_folder = '/content'

#similarities
similar_images = find_similar_images(target_image_path, dataset_folder)
image_size =(224,224)


print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Distance: {distance}")

plot_images(similar_images, target_image_path)
