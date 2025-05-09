# -*- coding: utf-8 -*-
"""paper ten.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F5tlyA9rB6Q5uoey8cQMdY3f4YGBSR_5
"""

!pip install tensorflow
import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Download dataset
path = kagglehub.dataset_download("fernando2rad/x-ray-lung-diseases-images-9-classes")
print("Dataset downloaded to:", path)

import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    classes = os.listdir(source_dir)

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        train_val, test = train_test_split(images, test_size=1 - (train_ratio + val_ratio), random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        for split_name, split_data in zip(['train', 'val', 'test'], [train, val, test]):
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_data:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy(src, dst)

# Usage:
processed_source = '/kaggle/input/x-ray-lung-diseases-images-9-classes'
processed_output = '/kaggle/working/xray_split'
split_dataset(processed_source, processed_output)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generators
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(processed_output, 'train'),
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(processed_output, 'val'),
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(processed_output, 'test'),
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate

# Step 1: Adapt grayscale input for ResNet50 (expects 3 channels)
# Convert grayscale to 3-channel by duplicating the single channel
def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

train_generator = train_generator
val_generator = val_generator
test_generator = test_generator

# Step 2: Build modified ResNet50 model
IMG_SIZE = (256, 256)
NUM_CLASSES = train_generator.num_classes
INIT_LR = 1e-4
EPOCHS = 10

# Input shape must match 3 channels for ResNet50
input_tensor = Input(shape=(256, 256, 1))
x = tf.keras.layers.Lambda(grayscale_to_rgb)(input_tensor)  # Convert grayscale to RGB

# Load base ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

# Freeze or fine-tune layers
for layer in base_model.layers:
    layer.trainable = True  # Fine-tuning
# Add custom head
head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(256, activation='relu')(head)
head = Dropout(0.5)(head)
output = Dense(NUM_CLASSES, activation='softmax')(head)

model = Model(inputs=input_tensor, outputs=output)

# Step 3: Compile model
opt = Adam(learning_rate=INIT_LR)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Step 4: Callbacks
checkpoint = ModelCheckpoint("best_resnet50_xray.h5", monitor='val_accuracy', save_best_only=True, mode='max')
earlystop = EarlyStopping(patience=5, restore_best_weights=True)

# Step 5: Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# Step 6: Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Step 7: Confusion matrix & report
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Predict and get labels
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import cv2
import tensorflow.keras.backend as K

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Choose a sample image
sample_img, _ = test_generator[0]  # First batch
img = sample_img[0:1]  # First image

# Generate heatmap
heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name='conv5_block3_out')

# Overlay heatmap
img_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img)).numpy()[0]
heatmap_resized = cv2.resize(heatmap, (256, 256))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed = heatmap_colored * 0.4 + (img_rgb * 255)

plt.imshow(np.uint8(superimposed))
plt.axis('off')
plt.title("Grad-CAM")
plt.show()

!pip install lime

from lime import lime_image
from skimage.segmentation import mark_boundaries

explainer = lime_image.LimeImageExplainer()

# Prepare image (convert grayscale to RGB for LIME)
image = test_generator[0][0][0]
rgb_image = tf.image.grayscale_to_rgb(tf.expand_dims(image, axis=0))[0].numpy()

# Define prediction function for LIME
def predict_fn(images):
    images = np.array(images)
    images = tf.image.rgb_to_grayscale(images)
    return model.predict(images)

explanation = explainer.explain_instance(
    rgb_image.astype('double'),
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

# Show explanation
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    hide_rest=False,
    num_features=10,
    min_weight=0.0
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title("LIME Explanation")
plt.axis('off')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Get all features and predictions
features = []
predictions = []
labels = []

for i in range(len(test_generator)):
    batch_x, batch_y = test_generator[i]
    features.append(batch_x.reshape(batch_x.shape[0], -1))
    preds = model.predict(batch_x)
    predictions.append(np.argmax(preds, axis=1))
    labels.append(np.argmax(batch_y, axis=1))

X = np.vstack(features)
y_true = np.hstack(labels)
y_pred = np.hstack(predictions)

# Reduce dimensions
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# Train surrogate model
surrogate = DecisionTreeClassifier(max_depth=5)
surrogate.fit(X_reduced, y_pred)

print("Surrogate Accuracy:", accuracy_score(y_true, surrogate.predict(X_reduced)))

from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(surrogate, filled=True, max_depth=2, fontsize=10)
plt.title("Surrogate Model (Decision Tree)")
plt.show()