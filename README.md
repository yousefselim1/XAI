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
    - **Pretrained Models**: Models like **VGG16**, **ResNet**, and **InceptionV3** can be used for transfer learning, which are fine-tuned to detect lung diseases with high accuracy.
    - **Variational Models**: Examples include **Variational Autoencoders (VAEs)** or **GAN-based architectures**, which are capable of learning the latent representations of lung disease images and can improve classification by generating new synthetic data for training.

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

Image Preprocessing and Feature Extraction
The notebook implements the following image preprocessing techniques:

Histogram Equalization: Improves the contrast of images, making important features in low-contrast images more visible.

Edge Detection:

Sobel Filter: Detects edges in the image, useful for recognizing structural patterns like boundaries and transitions.

Canny Edge Detection: A popular edge-detection algorithm used for identifying sharp changes in intensity.
