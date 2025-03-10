package ml.DigitclassifierMLP;

public class TwoFoldTestMlp {
    private MLPAlgorithm mlp; // This is the MLP (neural network) model we will train and test

    // This constructor  initialises the MLP that we will use for two-fold validation
    public TwoFoldTestMlp(MLPAlgorithm mlp) {
        this.mlp = mlp; // Thus stores the MLP so we can use it for training and testing
    }

    // This method runs two-fold validation on two data-sets
    public void runTwoFoldValidation(double[][] dataSet1, double[][] dataSet2) {
        // Fold 1: Trains on dataSet1, test on dataSet2
        System.out.println("Running Fold 1...");
        double fold1Accuracy = trainAndTest(dataSet1, dataSet2); //Train on dataSet1, test on dataSet2
        System.out.println("Accuracy: " + fold1Accuracy + "%"); // This Prints the accuracy for fold 1

        // Fold 2: Trains on dataSet2, test on dataSet1 ,we swap the roles
        System.out.println("Running Fold 2...");
        double fold2Accuracy = trainAndTest(dataSet2, dataSet1); // Train on dataSet2, test on dataSet1
        System.out.println("Accuracy: " + fold2Accuracy + "%"); // This prints the accuracy for fold 2

        // This calculates the overall accuracy as the average of fold 1 and fold 2
        double overallAccuracy = (fold1Accuracy + fold2Accuracy) / 2; 
        System.out.println("Average Accuracy: " + overallAccuracy + "%"); //This prints the overall average accuracy
    }

    // This method trains the MLP on the training set and tests it on the testing set
    private double trainAndTest(double[][] trainSet, double[][] testSet) {
        // Train the model for multiple epochs
        for (int epoch = 0; epoch < 300; epoch++) { // Run 300 epochs to train the model properly
            for (double[] data : trainSet) {
                double[] inputs = new double[data.length - 1]; // Input features for all columns except the last one
                double[] targets = new double[10]; // Target is a one-hot encoded array for 10 possible classes (0-9)
                
                // Copies input features from data for all but the last column
                System.arraycopy(data, 0, inputs, 0, data.length - 1); 

                // Get the label (class) from the last column
                int label = (int) data[data.length - 1]; 
                targets[label] = 1.0; // Convert the label into a one-hot encoded target (e.g., if label=3, targets[3] = 1)

                //This updates the MLP weights using back-propagation
                mlp.backpropagation(inputs, targets); 
            }
        }

        // This will test the model's performance on the testing set
        int correct = 0; // This counts the number of correct predictions
        for (double[] data : testSet) {
            double[] inputs = new double[data.length - 1]; // Input features for all but the last column
            
            // Copies input features from data for all but the last column
            System.arraycopy(data, 0, inputs, 0, data.length - 1); 

            // Gets the actual label for the correct class from the last column
            int label = (int) data[data.length - 1]; 

            // Get the predicted output from the MLP
            double[] outputs = mlp.forwardPass(inputs); 
            
            // This finds the class with the highest predicted value (argMax)
            int predictedLabel = argMax(outputs); 

            // This checks if the predicted label matches the actual label
            if (predictedLabel == label) {
                correct++; // This Count it as a correct prediction
            }
        }

        // This calculates accuracy as the percentage of correct predictions
        return (100.0 * correct) / testSet.length; //This will return accuracy as a percentage
    }

    // This method returns the index of the maximum value in an array which basically, finds which class has the highest probability
    private int argMax(double[] array) {
        int index = 0; // Start by assuming the max value is at index 0
        double max = array[0]; // Set the initial max value to the first element in the array
        for (int currentIndex = 1; currentIndex < array.length; currentIndex++) {
            if (array[currentIndex] > max) { // If we find a value larger than the current max
                max = array[currentIndex]; // This will Update the max value
                index = currentIndex; // This updates the index to the new max position
            }
        }
        return index; //It will return the index of the largest value
    }
}

