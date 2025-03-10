package ml.digitclassifierKnn;

public class KNNAlgorithm {
    private final int kNeighbors; // This is how many nearest neighbours we want to look at k value

    // Constructor to set how many neighbours we want to consider k
    public KNNAlgorithm(int kValue) {
        this.kNeighbors = kValue; // This stores the k value so we can use it later
    }

    // This method classifies a test instance based on the k-nearest neighbours from the training data
    public int classify(double[] testInstance, double[][] trainingData) {
        // These arrays will hold the closest k neighbours and their distances from the test instance
        double[][] nearestNeighbors = new double[kNeighbors][trainingData[0].length]; 
        double[] distances = new double[kNeighbors]; // This is the Distance to the k closest neighbours

        // This initialises the distances array with the largest possible value so anything we calculate will be smaller
        for (int index = 0; index < kNeighbors; index++) {
            distances[index] = Double.MAX_VALUE; // This sets the initial distances to max value
        }

        //This loops through every instance in the training data to find the k-nearest neighbours
        for (double[] trainInstance : trainingData) {
            double distance = euclideanDistance(testInstance, trainInstance); // This calculates the distance between test instance and current training instance

            // We want to replace the worst farthest neighbour if the new one is closer
            int maxIndex = 0; // This will track which of the current k-neighbors is the farthest
            for (int index = 1; index < kNeighbors; index++) {
                if (distances[index] > distances[maxIndex]) { 
                    maxIndex = index; // This updates maxIndex to the index of the farthest neighbour
                }
            }

            // If the new distance is smaller than the current worst neighbour, it will replace it
            if (distance < distances[maxIndex]) {
                distances[maxIndex] = distance; //This Updates the distance to the new closer neighbour
                nearestNeighbors[maxIndex] = trainInstance.clone(); // Store the entire training instance as a nearest neighbour
            }
        }

        // We now know the k-nearest neighbours; now, we must vote to forecast the class of the test instance

        int numClasses = 10; // There are 10 possible classes for classification digits 0-9. 
     // Each class represents one of the digits from 0 to 9. 
     // We need this so we can track how many votes each class gets from the k-nearest neighbours.

        double[] votes = new double[numClasses]; // Each index represents a class, and we'll count votes for each

        // Each of the k neighbours gets to vote for its class label
        for (double[] neighbor : nearestNeighbors) {
            int label = (int) neighbor[neighbor.length - 1]; // The last value in the neighbour array is its label class
            double weight = 1.0 / Math.pow(euclideanDistance(testInstance, neighbor) + 1e-3, 4.5); 
            // The closer the neighbour, the higher its influence (weight) on the vote
            votes[label] += weight; // Add the weight to the total votes for this label
        }

        //This finds which class got the most votes
        int predictedLabel = -1; // This stores the label of the class with the most votes
        double maxVotes = 0; // Store the number of votes for the most popular class
        for (int index = 0; index < votes.length; index++) {
            if (votes[index] > maxVotes) {
                maxVotes = votes[index]; // This updates max votes
                predictedLabel = index; // This updates the predicted label to this class
            }
        }

        // Return the predicted label (the one with the most votes)
        return predictedLabel;
    }

    // This calculates the Euclidean distance between two points
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        // This loops through each feature, but skips the last column since it's the label
        for (int featureIndex = 0; featureIndex < a.length - 1; featureIndex++) {
            sum += Math.pow(a[featureIndex] - b[featureIndex], 2); // (x1 - x2)^2 for each feature
        }
        return Math.sqrt(sum); // Return the square root of the sum the distance
    }
}
