package ml.DigitclassifierMLP;

import java.util.Random;

public class MLPAlgorithm {
    private int inputNodes; // This the number of input nodes 
    private int hiddenNodes; // This the number of nodes in the hidden layer
    private int outputNodes; // This the number of output nodes (for classification, like 10 for digits 0-9)
    private double[][] weightsInputHidden; // This is the weights connecting input layer to hidden layer
    private double[][] weightsHiddenOutput; // This is the Weights connecting hidden layer to output layer
    private double learningRate; // This controls how much weights get updated during training

    // Constructor to set up the MLP structure (input, hidden, and output nodes) and learning rate
    public MLPAlgorithm(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;

        // This Initialises the weights randomly to break symmetry and allow the model to learn
        initializeWeights();
    }

    // This method initialises weights randomly for both input-to-hidden and hidden-to-output connections
    private void initializeWeights() {
        Random random = new Random(); // The random number generator for random weight initialisation

        // This initialise weights for connections between input layer and hidden layer
        weightsInputHidden = new double[inputNodes][hiddenNodes];
        for (int columnIndex = 0; columnIndex < inputNodes; columnIndex++) {
            for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
                // Random weight between -0.5 and 0.5
                weightsInputHidden[columnIndex][hiddenNodeIndex] = random.nextDouble() - 0.5; 
            }
        }

        // This initialises weights for connections between hidden layer and output layer
        weightsHiddenOutput = new double[hiddenNodes][outputNodes];
        for (int columnIndex = 0; columnIndex < hiddenNodes; columnIndex++) {
            for (int outputNodeIndex = 0; outputNodeIndex < outputNodes; outputNodeIndex++) {
                // Random weight between -0.5 and 0.5
                weightsHiddenOutput[columnIndex][outputNodeIndex] = random.nextDouble() - 0.5; 
            }
        }
    }

 // ReLU (Rectified Linear Unit) activation function - which will set any negative values to 0 and leaves positive values unchanged.
 // This helps the network learn by only allowing positive signals to pass through.
    private double relu(double x) {
        return Math.max(0, x); // If x is negative, it becomes 0. If x is positive, it stays as x.
    
    }

    // Derivative of the ReLU function - returns 1 if x > 0, otherwise returns 0
    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    // This method performs a forward pass of the input through the MLP (input -> hidden -> output)
    public double[] forwardPass(double[] inputs) {
        double[] hiddenOutputs = new double[hiddenNodes]; // This stores the activations of hidden layer nodes

        // This calculates activations for each node in the hidden layer
        for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
            for (int inputNodeIndex = 0; inputNodeIndex < inputNodes; inputNodeIndex++) {
                hiddenOutputs[hiddenNodeIndex] += inputs[inputNodeIndex] * weightsInputHidden[inputNodeIndex][hiddenNodeIndex];
            }
            hiddenOutputs[hiddenNodeIndex] = relu(hiddenOutputs[hiddenNodeIndex]); // Apply ReLU to the sum
        }

        double[] finalOutputs = new double[outputNodes]; // This stores the activations of output layer nodes

        // This calculates activations for each node in the output layer
        for (int outputNodeIndex = 0; outputNodeIndex < outputNodes; outputNodeIndex++) {
            for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
                finalOutputs[outputNodeIndex] += hiddenOutputs[hiddenNodeIndex] * weightsHiddenOutput[hiddenNodeIndex][outputNodeIndex];
            }
            finalOutputs[outputNodeIndex] = relu(finalOutputs[outputNodeIndex]); // Apply ReLU to the sum
        }

        return finalOutputs; //This returns the activations of the output layer
    }

    // This method updates weights using back-propagation
    public void backpropagation(double[] inputs, double[] targets) {
        double[] hiddenOutputs = new double[hiddenNodes]; // Activations for the hidden layer nodes

        // This forwards the pass to compute hidden layer activations
        for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
            for (int inputNodeIndex = 0; inputNodeIndex < inputNodes; inputNodeIndex++) {
                hiddenOutputs[hiddenNodeIndex] += inputs[inputNodeIndex] * weightsInputHidden[inputNodeIndex][hiddenNodeIndex];
            }
            hiddenOutputs[hiddenNodeIndex] = relu(hiddenOutputs[hiddenNodeIndex]); // Apply ReLU to the sum
        }

        // This forwards the pass to compute output layer activations
        double[] finalOutputs = forwardPass(inputs); 

        // This calculates the error for the output layer
        double[] outputErrors = new double[outputNodes];
        for (int outputNodeIndex = 0; outputNodeIndex < outputNodes; outputNodeIndex++) {
            outputErrors[outputNodeIndex] = targets[outputNodeIndex] - finalOutputs[outputNodeIndex]; 
            // Error = Target - Prediction
        }

        // This calculates the error for the hidden layer
        double[] hiddenErrors = new double[hiddenNodes];
        for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
            for (int outputNodeIndex = 0; outputNodeIndex < outputNodes; outputNodeIndex++) {
                hiddenErrors[hiddenNodeIndex] += outputErrors[outputNodeIndex] * weightsHiddenOutput[hiddenNodeIndex][outputNodeIndex];
            }
        }

        // This update the weights connecting hidden layer to output layer
        for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
            for (int outputNodeIndex = 0; outputNodeIndex < outputNodes; outputNodeIndex++) {
                weightsHiddenOutput[hiddenNodeIndex][outputNodeIndex] += 
                    learningRate * outputErrors[outputNodeIndex] * reluDerivative(finalOutputs[outputNodeIndex]) * hiddenOutputs[hiddenNodeIndex];
                // Weight change is based on error, ReLU derivative, and the hidden layer activation
            }
        }

        // This update the weights connecting input layer to hidden layer
        for (int inputNodeIndex = 0; inputNodeIndex < inputNodes; inputNodeIndex++) {
            for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodes; hiddenNodeIndex++) {
                weightsInputHidden[inputNodeIndex][hiddenNodeIndex] += 
                    learningRate * hiddenErrors[hiddenNodeIndex] * reluDerivative(hiddenOutputs[hiddenNodeIndex]) * inputs[inputNodeIndex];
                // Weight change is based on error, ReLU derivative, and the input value
            }
        }
    }
}


