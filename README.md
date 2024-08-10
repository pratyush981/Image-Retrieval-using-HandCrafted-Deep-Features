This project presents a comprehensive image retrieval system that integrates 
traditional image features, such as color histograms, with deep learning-based 
features extracted using the pre-trained VGG16 model. The goal is to leverage 
the complementary strengths of both methods: the detailed color information 
captured by histograms and the intricate, high-level features learned by deep 
neural networks.  
The methodology involves extracting color histograms to capture the 
distribution of colors within an image and deep features from the 'avg_pool' 
layer of VGG16 to encapsulate complex image content. These features are 
concatenated to form a unified feature vector, providing a comprehensive 
representation of each image. Similarity between images is measured using 
Euclidean distance on the concatenated feature vectors.
![output CBIR](https://github.com/user-attachments/assets/35dd3943-2559-4136-bb54-9b7b4bd37e0d)
