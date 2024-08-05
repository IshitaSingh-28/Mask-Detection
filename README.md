# Face Mask Detection Project

## Project Summary
This project focuses on developing a Convolutional Neural Network (CNN) to detect whether a person is wearing a mask or not using image data. The dataset contains images of people with and without masks, and the model is trained to classify these images. The project involves image preprocessing, data augmentation, model building, training, evaluation, and prediction using a deep learning framework.

## Technologies Used
- **Python**: The programming language used for data processing and model development.
- **Kaggle API**: For downloading the dataset.
- **NumPy**: For numerical operations on image data.
- **Matplotlib**: For visualizing images and plotting the training history.
- **OpenCV**: For image processing and reading images for predictions.
- **PIL (Python Imaging Library)**: For image manipulation such as resizing and converting images.
- **Scikit-learn**: For splitting the dataset into training and testing sets.
- **TensorFlow & Keras**: For building and training the CNN model.
- **Google Colab**: For running the Jupyter Notebook in an online environment with GPU support.

## Step-by-Step Explanation

### 1. Dataset Preparation
- **Download the Dataset**: Using Kaggle API, the face mask dataset is downloaded and extracted.
- **Load the Dataset**: The images are loaded from the extracted files, with separate directories for images with masks and without masks.

### 2. Data Preprocessing
- **Image Loading and Resizing**: Images are read using PIL, resized to 128x128 pixels, and converted to RGB format.
- **Label Creation**: Labels are created, where `1` represents images with masks and `0` represents images without masks.
- **Combine Data and Labels**: Images and their corresponding labels are combined into NumPy arrays.

### 3. Train-Test Split
- **Split Data**: The dataset is split into training and testing sets using `train_test_split` from Scikit-learn.
- **Data Scaling**: Image pixel values are scaled to a range of 0 to 1 by dividing by 255.

### 4. Building the CNN Model
- **Model Architecture**: 
  - **Conv2D and MaxPooling2D Layers**: Two sets of convolutional and max-pooling layers for feature extraction.
  - **Flatten Layer**: Flattens the 2D matrix data to a vector.
  - **Dense and Dropout Layers**: Fully connected layers with dropout for regularization.
  - **Output Layer**: A dense layer with sigmoid activation for binary classification.
- **Model Compilation**: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.
- **Model Training**: The model is trained on the training data with validation split.

### 5. Model Evaluation
- **Evaluate Model**: The model is evaluated on the test data to determine accuracy.
- **Plotting Results**: The training and validation loss and accuracy are plotted to visualize the training process.

### 6. Prediction System
- **Predictive System**: 
  - Takes an image path as input.
  - Reads and processes the image.
  - Uses the trained model to predict if the person in the image is wearing a mask.
  - Displays the prediction result.
