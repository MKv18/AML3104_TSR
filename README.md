# AML3104_TSR

# Traffic Sign Recognition
This project is a traffic sign recognition system that can identify and classify traffic signs from images. The system uses machine learning algorithms to detect and recognize traffic signs, and it can be used in various applications such as autonomous vehicles and advanced driver assistance systems (ADAS) in future.

# About the project
This project is a real-time traffic sign recognition system built using Python, OpenCV, and a pre-trained CNN model, capable of detecting and recognizing traffic signs from both video streams and images.

The project includes a Jupyter notebook with the following:

Data exploration and preprocessing
Model training and evaluation
Model deployment

# Data Sources
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data

# Dataset
The dataset used in this project is the German Traffic Sign Dataset. This dataset contains more than 50,000 images of traffic signs belonging to 43 different classes. The images are in color and have a resolution of 32x32 pixels. The dataset is divided into training, validation, and testing sets, with 39,209, 12,630, and 12,630 images, respectively.

The dataset is stored in the data directory of the Git repository. The data directory contains three subdirectories: training, validation, and testing. Each subdirectory contains the images for the corresponding dataset.

# Code
The code for this project is written in Python and is located in the FinalCode directory of the Git repository. The code is a Jupyter notebook that contains the following sections:

# Data Loading: This section loads the dataset from the data directory and preprocesses it for training and testing. The section uses the tensorflow library to load the images and their corresponding labels.

# Data Augmentation: This section applies data augmentation techniques to the training dataset to increase its size and diversity. The section uses the tensorflow library to apply random rotations, translations, and zooms to the images.

# Model Architecture: This section defines the neural network architecture used for traffic sign recognition. The architecture is based on the VGG16 model, which is a convolutional neural network (CNN) architecture that was proposed in 2014. The section uses the tensorflow and keras libraries to define and train the model.

# Training: This section trains the neural network model using the training dataset. The section uses the tensorflow library to train the model and save the trained model to a file.

# Testing: This section tests the neural network model using the testing dataset. The section loads the trained model from a file and uses it to classify the images in the testing dataset. The section uses the tensorflow library to load the model and classify the images.
