package network;

import config.Config;
import math.Matrix;

public class NeuralNetwork {
    private Matrix weightsHiddenInput;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasOutput;

    //initializes weights and biases randomly
    public NeuralNetwork(){
    weightsHiddenInput = new Matrix(Config.INPUT_SIZE, Config.HIDDEN_SIZE);
    weightsHiddenInput.randomize(-0.5, 0.5);

    weightsHiddenOutput = new Matrix(Config.HIDDEN_SIZE, Config.OUTPUT_SIZE);
    weightsHiddenOutput.randomize(-0.5, 0.5);

    biasHidden = new Matrix(1, Config.HIDDEN_SIZE);
    biasHidden.randomize(-0.5, 0.5);

    biasOutput = new Matrix(1, Config.OUTPUT_SIZE);
    biasOutput.randomize(-0.5, 0.5);
    }

    //returns output layer of neural network
    public Matrix forward(Matrix inputMatrix) {

        // hidden layer
        Matrix hiddenLayer = inputMatrix.mul(weightsHiddenInput);  //internal layer is gotten by multiplying input and hidden input matrix
        hiddenLayer = hiddenLayer.elementAdd(biasHidden);      //add the bias for calculation
        hiddenLayer = hiddenLayer.map(x -> Math.tanh(x));

        // output layer
        Matrix outputLayer = hiddenLayer.mul(weightsHiddenOutput);   //get the output layer by matrix multiplication of hiddenlayer with hidden output
        outputLayer = outputLayer.elementAdd(biasOutput);
        outputLayer = softmax(outputLayer);

        return outputLayer;
    }


    //takes 6 raw numbers and convert them into 6 probabilities that add up to 1
    //does that through the e^x function because it always oitputs a positive number no matter what
    //e^(-5)  =  0.0067
    //e^(0)   =  1.0
    //e^(5)   =  148.4
    private Matrix softmax(Matrix inputMatrix){
        Matrix resultMatrix = new Matrix(1, Config.OUTPUT_SIZE);    //makes a new horizontal vector
        double sum = 0.0;

        for (int i = 0; i < Config.OUTPUT_SIZE; i++) {
            sum += Math.exp(inputMatrix.get(0, i));       //get is from matrix class
        }

        for (int i = 0; i < Config.OUTPUT_SIZE; i++) {
            resultMatrix.set(0, i, Math.exp(inputMatrix.get(0, i)) / sum);      //divide each element by the sum
        }

        return resultMatrix;
    }
}
