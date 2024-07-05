# SVM Image Classification for Cats and Dogs

This project implements a Support Vector Machine (SVM) classifier to classify images of cats and dogs. The SVM model is trained on a dataset containing images of cats and dogs, and it learns to differentiate between the two classes based on features extracted from the images.

## Dataset
The dataset consists of images of cats and dogs collected from various sources. Each image is labeled as either a cat or a dog. The dataset is split into training and testing sets to facilitate model training and evaluation.

## Approach
1. **Preprocessing**: The images are preprocessed by resizing them to a uniform size and converting them to grayscale. This ensures that all images have the same dimensions and color representation.

2. **Feature Extraction**: Features are extracted from the preprocessed images to represent their characteristics. In this project, simple pixel intensity values are used as features. The images are flattened into a one-dimensional array, and each pixel value serves as a feature.

3. **Model Training**: The SVM classifier is trained on the extracted features from the training set. The SVM learns to find an optimal decision boundary that separates the feature space into regions corresponding to cats and dogs.

4. **Model Evaluation**: The trained SVM model is evaluated on the testing set to assess its performance. Various evaluation metrics such as accuracy, precision, recall, and F1-score are computed to measure the classifier's effectiveness in distinguishing between cats and dogs.

## Usage
1. **Download the Dataset**: Obtain the dataset containing images of cats and dogs. You can use publicly available datasets or collect your own images.

2. **Preprocess the Images**: Preprocess the images by resizing them to a uniform size and converting them to grayscale if necessary. Ensure that the images are organized into separate folders for cats and dogs.

3. **Train the SVM Model**: Run the provided Python script to train the SVM classifier on the preprocessed images. Adjust hyperparameters as needed to optimize the model performance.

4. **Evaluate the Model**: Evaluate the trained SVM model using the testing set to measure its accuracy and performance metrics. Use the classification report to analyze the model's performance in detail.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn

Install the required dependencies using pip:
```bash
pip install opencv-python numpy scikit-learn
```

## Example Output
After running the script, you will get an output similar to this:

```plaintext
              precision    recall  f1-score   support

         cat     1.0000    1.0000    1.0000         3
         dog     1.0000    1.0000    1.0000         3

    accuracy                         1.0000         6
   macro avg     1.0000    1.0000    1.0000         6
weighted avg     1.0000    1.0000    1.0000         6
```

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.