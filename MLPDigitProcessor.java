package ml.DigitclassifierMLP;

public class MLPDigitProcessor {
    public static void main(String[] args) {
        try {
            //This loads the data-sets using the DataLoaderMLP
            MLPDatasetLoader dataLoader = new MLPDatasetLoader(); // This is responsible for loading the CSV data files

            // This loads the first data-set from "dataSet1.csv"
            double[][] dataSet1 = dataLoader.loadData("dataSet1.csv");

            //This loads the second data-set from "dataSet2.csv"
            double[][] dataSet2 = dataLoader.loadData("dataSet2.csv");

            // This normalises both data-sets so that feature values are between 0 and 1
            normalizeData(dataSet1); // This normalises the first data-set
            normalizeData(dataSet2); // This normalises the second data-set

            // Define the structure and parameters of the MLP (Multi-Layer Perceptron)
            int inputNodes = 64; // This is the number of features (input neurons) in the input layer
            int hiddenNodes = 512; // This is the number of neurons in the hidden layer (more nodes can improve learning)
            int outputNodes = 10; // This is the number of output classes (like digits 0-9)
            double learningRate = 0.01; // Learning rate controls how big the weight updates are during training

            //This creates an instance of the MLP which basically, builds the neural network
            MLPAlgorithm mlp = new MLPAlgorithm(inputNodes, hiddenNodes, outputNodes, learningRate); 
            // This MLP will have 64 input nodes, 512 hidden nodes, and 10 output nodes

            // Set up two-fold cross-validation to test the MLP
            TwoFoldTestMlp evaluator = new TwoFoldTestMlp(mlp); 
            // The TwoFoldTestMlp will split the data-sets into training and testing sets, swapping them in the second fold

            // This will run the two-fold validation process to test the performance of the MLP
            evaluator.runTwoFoldValidation(dataSet1, dataSet2); 
            // In fold 1, dataset1 is used for training, dataset2 for testing. 
            // In fold 2, the roles of dataset1 and dataset2 are swapped.

        } catch (Exception e) {
            e.printStackTrace(); // If something goes wrong, it will print the error message
        }
    }

    // This method normalises all feature values in the data-set to a range of [0, 1]
    private static void normalizeData(double[][] data) {
        for (double[] row : data) {
            // This loops through each feature in the row, but leave out the label (last column)
            for (int columnIndex = 0; columnIndex < row.length - 1; columnIndex++) { 
                row[columnIndex] /= 255.0; //This divides each feature value by 255 to bring it to the range [0, 1]
            }
        }
    }
}


