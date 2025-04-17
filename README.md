# XAI

# **Lung Disease Detection Using X-ray Images**

This project leverages deep learning models for detecting and classifying lung diseases from chest X-ray images. The notebook applies advanced image preprocessing techniques such as **Histogram Equalization**, **Edge Detection** (Sobel and Canny), and utilizes various **deep learning models** (such as **Convolutional Neural Networks (CNNs)**, **Pretrained Models**, and **Variational Models**) for classifying lung diseases. Additionally, interpretable methods like **LIME**, **SHAP**, **Surrogate Models**, **Grad-CAM++**, and **Saliency Maps** are employed to enhance the transparency and explainability of the model’s predictions.

---

## **Features**

- **Image Preprocessing Techniques**:
    - **Histogram Equalization**: Improves the contrast and clarity of images.
    - **Sobel Edge Detection**: Highlights edges and boundaries within the image.
    - **Canny Edge Detection**: Identifies sharp changes in pixel intensity for boundary detection.

- **Deep Learning Models**:
    - **CNN** (Convolutional Neural Network): Used for detecting and classifying lung diseases from X-ray images.


- **Interpretability Methods**:
    - **Grad-CAM++**: Visualizes highlighted areas in the image that influence the model’s predictions by showing where the model is focusing during classification.
    - **LIME** (Local Interpretable Model-Agnostic Explanations): Provides local explanations for individual predictions by approximating the model locally with an interpretable surrogate model.
    - **SHAP** (SHapley Additive exPlanations): Quantifies the contribution of each feature (pixel) to the model’s decision, giving a clear view of how each part of the image affects the prediction.
    - **Surrogate Models**: Simpler, interpretable models (like decision trees) that approximate the behavior of the complex deep learning model, aiding explanation.
    - **Saliency Maps**: Highlights the most important pixels or regions in the image that have the largest influence on the model’s decision.

---

## **Dataset**

The dataset used for training and testing the model is the **X-ray Lung Diseases Dataset**. It contains images for multiple categories of lung diseases, including but not limited to:

- **Anatomia Normal**
- **Pneumonia**
- **COVID-19**
- **Pneumothorax**
- **And more...**

This dataset is downloaded using the **KaggleHub tool** from the dataset [X-ray Lung Diseases - 9 Classes](https://www.kaggle.com/fernando2rad/x-ray-lung-diseases-images-9-classes). The dataset consists of images in different categories, and our model uses these images to predict lung diseases.

```python
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("fernando2rad/x-ray-lung-diseases-images-9-classes")
print("Path to dataset files:", path)



```


## **Requirements**

To run this project, you need:

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Numpy
- Matplotlib
- Seaborn
- tqdm
- LIME
- SHAP
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install -r requirements.txt

```



## **Image Preprocessing and Feature Extraction**

The notebook implements the following image preprocessing techniques:

- **Histogram Equalization**: Improves the contrast of images, making important features in low-contrast images more visible.

### **Edge Detection**:
- **Sobel Filter**: Detects edges in the image, useful for recognizing structural patterns like boundaries and transitions.
- **Canny Edge Detection**: A popular edge-detection algorithm used for identifying sharp changes in intensity.

---

## **Model Overview**

The project implements the following models for detecting and classifying lung diseases from chest X-ray images:

### **Convolutional Neural Networks (CNNs)**:
CNNs are the primary model used in this project for classifying lung diseases. These models consist of multiple layers of convolutions and pooling, which allow them to learn hierarchical features of images, such as edges, textures, and complex structures.

- **Functionality**: The CNNs used in this project are designed to recognize patterns and detect lung abnormalities by learning from labeled chest X-ray images. 
- **Advantage**: CNNs are powerful for image classification tasks and help in automating the detection of lung diseases directly from raw images.

---

### **Pretrained Models (VGG16 and VGG19)**:
These are well-known Convolutional Neural Networks (CNNs) that have been pretrained on large image datasets like **ImageNet**. Pretraining allows the model to learn general features (like edges, textures, and basic shapes) from a large amount of data.

- **Fine-tuning**: In this project, these models are fine-tuned to specialize in detecting lung diseases, leveraging the learned features to classify new images. 
- **Approach**: By freezing most layers of the model and retraining only the final layers, pretrained models can be adapted for lung disease classification with fewer data and less training time.

---

### **Transfer Learning**:
Transfer learning is applied using pretrained models such as **VGG16** and **VGG19**. By using the weights from **ImageNet** and training the final layers on the lung disease dataset, the models can generalize well even with a smaller dataset of X-ray images.

- **Benefit**: Transfer learning improves model performance with less data and training time, making it particularly useful in medical image classification tasks where labeled data might be limited.

---

### **Variational Models (VAE and GANs)**:

#### **Variational Autoencoders (VAEs)**:
VAEs are generative models that learn to encode images into a compact representation and decode them back into images. They are used here to generate synthetic lung disease images, which helps in augmenting the dataset.

- **Purpose**: By generating new images that resemble the original dataset, VAEs provide more diverse training examples, improving the classification model's performance.

#### **Generative Adversarial Networks (GANs)**:
GANs consist of two networks: a generator that creates fake images and a discriminator that distinguishes fake images from real ones.

- **Purpose**: GANs help generate new synthetic X-ray images to improve model performance, particularly when the dataset is limited. The generated images provide valuable augmentation for training, making the model more robust.



