package ml.digitclassifierKnn;

public class KNNDigitProcessor {
    public static void main(String[] args) {
        
        // This is where it all starts the main method to run the KNN classification
        KNNDatasetLoader dataLoader = new KNNDatasetLoader(); // This is to load the data-sets we need for training and testing

        // Load the first data-set from the CSV file. 
        // This method reads the file, processes it, and returns the data in a 2D array
        double[][] dataSet1 = dataLoader.loadData("dataSet1.csv"); 

        // Load the second data-set from another CSV file. 
        // We now have two data-sets ready to be used for training and testing
        double[][] dataSet2 = dataLoader.loadData("dataSet2.csv"); 

        // Create a KNN classifier with k = 3 can be changed to see how it affects the accuracy
        KNNAlgorithm knn = new KNNAlgorithm(3); 

        // Set up the evaluator to test the KNN model using two-fold cross-validation
        // Basically, it splits the data-sets into training and testing sets, then swaps them to test twice
        TwoFoldTestKnn evaluator = new TwoFoldTestKnn(knn); 

        //This runs the two-fold validation. This is where the model actually gets trained and tested.
        evaluator.runTwoFoldValidation(dataSet1, dataSet2); 
    }
}

