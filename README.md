# XAI

Lung Disease Detection Using X-ray Images
This project leverages deep learning models for detecting and classifying lung diseases using chest X-ray images. The notebook applies advanced image preprocessing techniques such as Histogram Equalization, Edge Detection (Sobel and Canny), and uses Convolutional Neural Networks (CNNs) for classifying lung diseases. Additionally, interpretable methods such as LIME, SHAP, Surrogate Models, Grad-CAM++, and Saliency Maps are employed to enhance the transparency and explainability of the model's predictions.

Features
Image preprocessing using Histogram Equalization, Sobel Edge Detection, and Canny Edge Detection.

Deep Learning model (CNN) for lung disease classification from X-ray images.

Use of Grad-CAM++, LIME, SHAP, Surrogate Models, and Saliency Maps for interpretability.

Visualization of model predictions with highlighted areas using Grad-CAM++ and LIME explanations.

Dataset
The dataset used for training and testing the model is the X-ray Lung Diseases Dataset. It contains images for multiple categories of lung diseases including, but not limited to:

Anatomia Normal

Pneumonia

COVID-19

Pneumothorax

and more...

This dataset is downloaded using the KaggleHub tool from X-ray Lung Diseases - 9 Classes. The dataset consists of images in different categories, and our model uses these images to predict lung diseases.

import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("fernando2rad/x-ray-lung-diseases-images-9-classes")
print("Path to dataset files:", path)


Requirements
To run this project, you need:

Python 3.x

TensorFlow / Keras

OpenCV

Numpy

Matplotlib

Seaborn

tqdm

LIME

SHAP

Scikit-learn

You can install the required packages using the following command:
pip install -r requirements.txt


Setup
Clone this repository:
git clone https://github.com/your_username/lung-disease-detection.git

Install required dependencies:
pip install -r requirements.txt

Upload your dataset to the repository or point to the directory where the dataset is stored.

Image Preprocessing and Feature Extraction
The notebook implements the following image preprocessing techniques:

Histogram Equalization: This technique improves the contrast of images, making important features in low-contrast images more visible.

Edge Detection:

Sobel Filter: Detects edges in the image, useful for recognizing structural patterns like boundaries and transitions.

Canny Edge Detection: A popular edge-detection algorithm used for identifying sharp changes in intensity.

Model Overview
The model is based on Convolutional Neural Networks (CNNs), designed to classify chest X-ray images into different disease categories like Normal, Pneumonia, COVID-19, etc.

The model uses the preprocessed images (with histogram equalization and edge detection) for training.

Interpretable AI (XAI) Methods Used
To enhance the transparency and explainability of the deep learning model's predictions, several interpretable AI techniques have been implemented:

1. LIME (Local Interpretable Model-Agnostic Explanations)
LIME is used to explain individual predictions of the model by approximating the model locally with an interpretable surrogate model.

This technique highlights which parts of the image contributed to the model's classification.

LIME helps in understanding model decisions and is particularly useful for image classification models.

2. SHAP (SHapley Additive exPlanations)
SHAP values are calculated to explain the contribution of each feature (pixel in the image) to the model's decision.

SHAP provides a global view of the model by measuring the importance of each feature across all predictions and also offers a local explanation for individual predictions.

3. Surrogate Models
Surrogate models are simpler models that approximate the behavior of the complex deep learning model. These are typically decision trees or linear models that are more interpretable and can be used to explain the decision-making process of the deep neural network.

4. Grad-CAM++ (Gradient-weighted Class Activation Mapping)
Grad-CAM++ is used to visualize the important regions in the X-ray image that contributed to the model's decision.

It highlights the areas of the image that the model focused on when making the classification decision, making the model more transparent.

5. Saliency Maps
Saliency maps are gradient-based techniques that highlight the pixels in the image that are most responsible for the model's decision.

It shows the areas of the image that contribute to the high prediction confidence, making it easier to understand which parts of the image influence the model's output.

Model Evaluation
The model is evaluated using various metrics:

Accuracy: Measures the proportion of correct predictions.

Confusion Matrix: Shows the performance of the model in terms of True Positives, False Positives, True Negatives, and False Negatives.

Grad-CAM Visualization: Used to highlight the most important regions of the image for classification.

SHAP and LIME: Used to explain individual predictions, showing which features (or pixels) in the image were most important for the model's decision.

How to Run the Notebook
Open the notebook (lung_disease_detection.ipynb) in Google Colab or Jupyter.

Execute the cells sequentially. The notebook will process images, train the model, and visualize the results using the interpretability techniques mentioned.

The notebook also visualizes Grad-CAM and LIME explanations for predictions.

Results and Metrics
The model's performance on the test data is evaluated using:

Accuracy

Precision

Recall

F1-score

Grad-CAM visualizations

LIME and SHAP explanations

Usage
python lung_disease_classification.py

This will run the entire pipeline, from image preprocessing to model evaluation.


Contributing
If you would like to contribute to this project, please fork the repository and create a pull request. Make sure to add new features or fixes along with test cases.

