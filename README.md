Handwritten Digit Classification using KNN and MLP

Overview

This project implements machine learning algorithms from scratch to classify handwritten digits using the Optical Recognition of Handwritten Digits Dataset from the UCI Machine Learning Repository. The system was tested using two-fold cross-validation, as required in the coursework.

Dataset

Source: UCI Machine Learning Repository

Files Used: dataSet1.csv, dataSet2.csv

Description: The dataset contains pixel-based feature representations of handwritten digits (0-9), with 64 features per instance.

Objective

The goal of this coursework was to develop and compare different machine learning algorithms for digit classification. Two algorithms were implemented:

K-Nearest Neighbours (KNN)

Multi-Layer Perceptron (MLP)

Their performance was evaluated based on accuracy across two validation folds.

Methodology

1. K-Nearest Neighbours (KNN)

k-value: Set to 3

Distance Metric: Euclidean distance

Weighted Voting: Neighbours' contributions were weighted by the inverse of their distance raised to a power of 4.5.

Normalization: Feature values were normalized for consistency.

Validation Method: Two-fold cross-validation.

Results:

Fold 1 Accuracy: 98.04%

Fold 2 Accuracy: 98.65%

Average Accuracy: 98.35%

2. Multi-Layer Perceptron (MLP)

Network Architecture:

Input Layer: 64 nodes (one for each feature)

Hidden Layer: 512 nodes (ReLU activation)

Output Layer: 10 nodes (one for each digit class)

Learning Rate: 0.01

Weight Initialization: Random initialization at the start of training.

Training Method: Backpropagation with gradient descent.

Validation Method: Two-fold cross-validation.

Results:

Fold 1 Accuracy: 97.65%

Fold 2 Accuracy: 99.47%

Average Accuracy: 98.56%

Algorithm Comparison

Algorithm

Fold 1 Accuracy

Fold 2 Accuracy

Average Accuracy

KNN

98.04%

98.65%

98.35%

MLP

97.65%

99.47%

98.56%

Key Observations

MLP achieved slightly higher accuracy due to its ability to model complex relationships.

KNN provided consistent results with a simpler approach and no training phase.

MLP requires more computational resources and longer execution time compared to KNN.

The random weight initialization in MLP led to slight variations between runs.

Feedback from the instructor highlighted that the second fold in MLP consistently outperformed the first, indicating an issue with the test setup.

Potential Improvements

Implement additional classifiers such as Support Vector Machines (SVM) or Decision Trees.

Conduct hyperparameter tuning to further optimize performance.

Introduce confusion matrices to analyze misclassifications in more detail.

Improve variable naming and use of named constants to enhance code clarity.

Results and Achievements

Successfully implemented KNN and MLP from scratch.

Achieved high classification accuracy (~98%) for handwritten digits.

Provided valuable insights into algorithm performance and trade-offs.

Final Mark: 62/100

Future Work

Implementing additional ML models for improved accuracy.

Exploring dropout regularization to enhance MLP generalization.

Expanding the dataset for better real-world applicability.

Author

Yasin LesterCST3170 - Machine Learning Coursework
