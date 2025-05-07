# 🐾 Animal Classification Project

## 📘 Overview
This Jupyter Notebook (`Animal_project.ipynb`) contains a complete pipeline for building and training an animal image classifier using **TensorFlow** and the **ResNet50** architecture. The project demonstrates how to:

- ✅ Split a dataset into training and test sets  
- 🧼 Preprocess and load image data  
- 🧠 Build a transfer learning model using ResNet50  
- 🚀 Train and evaluate the model  
- 💾 Save the trained model for future use  
- 🔍 Make predictions on new images  

The classifier is designed to recognize **15 different animal classes**, including:
> Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

---

## 📦 Requirements

Make sure the following are installed in your environment:

- Python 3.x  
- TensorFlow 2.x  
- Keras  
- OpenCV  
- NumPy  
- Other standard Python libraries: `pandas`, `matplotlib`, etc.

---

## 📂 Dataset Requirements

The code expects an animal image dataset with the following structure:
- The `dataset/` directory should contain one subfolder for each animal class.
- Each subfolder must be named after the animal class (e.g., `Bear`, `Bird`, `Cat`, etc.).
- Inside each subfolder, include multiple image files representing that class (e.g., `bear_image_1.jpg`, `bear_image_2.jpg`, etc.).


**Key characteristics:**
- Images organized in subfolders by class
- Supported formats: JPG, JPEG, PNG
- Recommended size: **224x224 pixels** (auto-resized if needed)
- Balanced class distribution improves performance

---

## 🚀 How to Use

1. Place your dataset inside the `dataset/` folder as described above.  
2. Open and run `Animal_project.ipynb` sequentially:
   - 📁 Dataset split into train/test sets  
   - 🧠 Model construction, training, evaluation  
   - 💾 Model saved as `.h5` and `.keras`  
3. Use the provided prediction function to classify new images.

---

## 📊 Model Performance

- **Training Accuracy:** ~100%  
- **Validation Accuracy:** ~96.7%  

---

## 💾 Saved Models

Two formats of the trained model are saved:

- `animal_classifier_resnet50.h5` – Traditional HDF5 format  
- `animal_classifier_resnet50.keras` – Recommended modern Keras format  

---

## 🔍 Prediction Example

Use the built-in function to predict on a new image:
 and replace it with the location of the image to be tested.
