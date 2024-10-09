# Image Labeling Project

## Introduction
This project focuses on automatically labeling clothing images from a catalog using machine learning algorithms. We use two primary algorithms:
- **K-means**: For unsupervised classification to identify the predominant colors in an image.
- **K-Nearest Neighbors (K-NN)**: For supervised classification to label the type of clothing based on image features.

The system processes a given image and returns a label for the type of clothing and one or more labels for the colors.

## Project Structure

### Files
1. `images/`
   - **Train**: Training set of images with labeled data.
   - **Test**: Test set of images for experimentation.
   - **gt.json**: JSON file containing information about the class each image belongs to in the train and test sets.
   
2. **Algorithms**:
   - `Kmeans.py`: Implementation of the K-means algorithm to identify predominant colors in the images.
   - `KNN.py`: Implementation of the K-NN algorithm for clothing type classification.
   
3. **Tests**:
   - `TestCases_kmeans.py`: Unit tests for verifying the functionality of the K-means algorithm.
   - `TestCases_knn.py`: Unit tests for verifying the K-NN algorithm.

4. **Utilities**:
   - `utils.py`: Helper functions for color conversion and processing.
   - `utils_data.py`: Functions for result visualization and analysis.

5. **Final Labeling**:
   - `my_labeling.py`: Combines both K-means and K-NN results to label the images with clothing type and color.

### Dependencies
- Python 3.8+
- Numpy
- Scipy
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/leonhrt/AI-Product-Labelling.git
   cd AI-Product-Labelling
   ```

### Running the Project
1. To test the algorithms:
   ```bash
   python TestCases_kmeans.py
   python TestCases_knn.py
   ```
2. To see multiple statistics of the algorithms with the provided catalog:
   ```bash
   python my_labeling.py
   ```

### Algorithm Breakdown
- **K-means**:
   - Groups image pixels based on color similarity.
   - Returns the dominant colors in the image as RGB values.
   - These values are converted into labels using predefined color names.

- **K-NN**:
   - Classifies the type of clothing using pixel intensity as features.
   - Uses distance measures to find the closest matching clothing type from the training set.

### Improvements and Analysis
- Experiment with different initializations and distance metrics in K-means and K-NN.
- Evaluate performance based on metrics like accuracy, convergence time, and inter-class distances.

### Results
Results are visualized using:
- **Qualitative Analysis**: Comparing images based on shape and color labels.
- **Quantitative Analysis**: Accuracy of clothing type and color classification.
