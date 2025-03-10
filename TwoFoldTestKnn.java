package ml.digitclassifierKnn;

public class TwoFoldTestKnn {
    private final KNNAlgorithm algorithm; // This holds the KNN algorithm that we will use to classify the data

    // This sets up the two-fold validation using the KNN algorithm we pass in
    public TwoFoldTestKnn(KNNAlgorithm algorithm) {
        this.algorithm = algorithm; // This stores the KNN algorithm so we can use it later
    }

    // This method runs two-fold cross-validation using two data-sets
    public void runTwoFoldValidation(double[][] dataSet1, double[][] dataSet2) {
        // First fold: uses dataSet1 for training and dataSet2 for testing
        System.out.println("Running Fold 1...");
        double accuracy1 = evaluate(dataSet1, dataSet2); // Train on dataSet1, test on dataSet2
        System.out.printf("Accuracy: %.6f%%%n", accuracy1); // This prints the accuracy for fold 1

        // Second fold: uses dataSet2 for training and dataSet1 for testing 
        System.out.println("Running Fold 2...");
        double accuracy2 = evaluate(dataSet2, dataSet1); // Train on dataSet2, test on dataSet1
        System.out.printf("Accuracy: %.6f%%%n", accuracy2); // Thus prints the accuracy for fold 2

        // Calculate the average accuracy from both folds
        double averageAccuracy = (accuracy1 + accuracy2) / 2.0; 
        System.out.printf("Average Accuracy: %.6f%%%n", averageAccuracy); // This will print the final average accuracy
    }

    // This method evaluates the model using a training data-set and a testing data-set
    private double evaluate(double[][] trainingData, double[][] testingData) {
        int correctPredictions = 0; // This will count how many predictions we get right

        // This will loop through every instance in the testing data to see if the model predicts it correctly
        for (double[] testInstance : testingData) {
            int predictedLabel = algorithm.classify(testInstance, trainingData); // Uses KNN to classify the test instance
            int actualLabel = (int) testInstance[testInstance.length - 1]; // The actual label is the last value in the row
            if (predictedLabel == actualLabel) {
                correctPredictions++; // If the predicted label matches the actual label, it will count it as correct
            }
        }

        // Calculates the accuracy as the percentage of correct predictions
        return ((double) correctPredictions / testingData.length) * 100; // This return accuracy as a percentage
    }
}


