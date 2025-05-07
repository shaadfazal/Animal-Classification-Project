# Animal Classification Project
Overview
This Jupyter Notebook (Animal_project.ipynb) contains a complete pipeline for building and training an animal image classifier using TensorFlow and the ResNet50 architecture. The project demonstrates how to:

Split a dataset into training and test sets

Preprocess and load image data

Build a transfer learning model using ResNet50

Train and evaluate the model

Save the trained model for future use

Make predictions on new images

The classifier is designed to recognize 15 different animal classes including bear, bird, cat, cow, deer, dog, dolphin, elephant, giraffe, horse, kangaroo, lion, panda, tiger, and zebra.

Requirements
To run this notebook, you'll need:

Python 3.x

TensorFlow 2.x

Keras

OpenCV

NumPy

Other standard Python data science libraries (Pandas, Matplotlib, etc.)

Dataset Requirements
The code expects an animal image dataset structured in the following way:

dataset/
    ├── Bear/
    │   ├── bear_image_1.jpg
    │   ├── bear_image_2.jpg
    │   └── ...
    ├── Bird/
    │   ├── bird_image_1.jpg
    │   ├── bird_image_2.jpg
    │   └── ...
    └── ... (other animal classes)
Key Dataset Characteristics:
Images organized in subfolders by class

Supported image formats (JPG/JPEG/PNG)

Recommended image size: 224x224 pixels (though the code will resize images)

Balanced distribution across classes is ideal for best performance

How to Use
Place your animal image dataset in the dataset/ folder with the structure shown above

Run the notebook cells sequentially:

First cell: Dataset splitting into train/test sets

Subsequent cells: Model building, training, and evaluation

The trained model will be saved as animal_classifier_resnet50.h5 and .keras formats

Use the prediction function to classify new animal images

Model Performance
The model achieves:

Training accuracy: ~100%

Validation accuracy: ~96.7%

Saved Models
The notebook saves two versions of the trained model:

animal_classifier_resnet50.h5 - Traditional HDF5 format

animal_classifier_resnet50.keras - Recommended Keras format

Prediction Example
The notebook includes a function to make predictions on new images:

python
predict_image("test/Kangaroo/Kangaroo_16_3.jpg")
This will output the predicted class and confidence score.
