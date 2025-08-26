import cv2 
import numpy as np 
import os  
from skimage.feature import local_binary_pattern 
from tensorflow.keras.applications import VGG16   
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances 
import matplotlib.pyplot as plt

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
 
def calculate_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def calculate_lbp_features(image_path, radius=1, n_points=8):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} not found or could not be opened.")

    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def calculate_deep_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    return vgg16_feature.flatten()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img_path))
    return images

def find_similar_images_combined(target_image_path, dataset_folder, num_similar=7):
    target_hist = calculate_color_histogram(target_image_path)
    target_lbp = calculate_lbp_features(target_image_path)
    target_deep = calculate_deep_features(target_image_path)

    target_features = np.concatenate([target_hist, target_lbp, target_deep])

    images = load_images_from_folder(dataset_folder)

    features_list = []
    for filename, img_path in images:
        hist = calculate_color_histogram(img_path)
        lbp = calculate_lbp_features(img_path)
        deep = calculate_deep_features(img_path)

        combined_features = np.concatenate([hist, lbp, deep])
        features_list.append((filename, img_path, combined_features))
 
    target_features = target_features.reshape(1, -1)
    distances = []
    for filename, img_path, features in features_list:
        features = features.reshape(1, -1)
        distance = euclidean_distances(target_features, features)[0][0]
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

target_image_path = '/content/330.jpg' 
dataset_folder = '/content'

similar_images = find_similar_images_combined(target_image_path, dataset_folder, num_similar=7)

print("Similar Images:")
for filename, img_path, distance in similar_images:
    print(f"Filename: {filename}, Path: {img_path}, Distance: {distance}")

plot_images(similar_images, target_image_path)
