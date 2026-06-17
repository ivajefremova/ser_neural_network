package network;

import config.Config;
import math.Matrix;

public class NeuralNetwork {
    private Matrix weightsInputHidden1;
    private Matrix weightsHidden1Hidden2;
    private Matrix weightsHidden2Output;
    private Matrix biasHidden1;
    private Matrix biasHidden2;
    private Matrix biasOutput;
    private Matrix lastInput;
    private Matrix lastHidden1;
    private Matrix lastHidden2;

    //getters and setters for model saver
    public Matrix getWeightsInputHidden1()   { return weightsInputHidden1; }
    public Matrix getWeightsHidden1Hidden2() { return weightsHidden1Hidden2; }
    public Matrix getWeightsHidden2Output()  { return weightsHidden2Output; }
    public Matrix getBiasHidden1()           { return biasHidden1; }
    public Matrix getBiasHidden2()           { return biasHidden2; }
    public Matrix getBiasOutput()            { return biasOutput; }

    public void setWeightsInputHidden1(Matrix m)   { weightsInputHidden1 = m; }
    public void setWeightsHidden1Hidden2(Matrix m) { weightsHidden1Hidden2 = m; }
    public void setWeightsHidden2Output(Matrix m)  { weightsHidden2Output = m; }
    public void setBiasHidden1(Matrix m)           { biasHidden1 = m; }
    public void setBiasHidden2(Matrix m)           { biasHidden2 = m; }
    public void setBiasOutput(Matrix m)            { biasOutput = m; }

    //initializes weights with Xavier init: range = sqrt(6 / (fan_in + fan_out))
    //keeps gradients from shrinking across tanh layers; biases start at 0
    public NeuralNetwork() {
        double xavierL1 = Math.sqrt(6.0 / (Config.INPUT_SIZE    + Config.HIDDEN_SIZE_1));
        double xavierL2 = Math.sqrt(6.0 / (Config.HIDDEN_SIZE_1 + Config.HIDDEN_SIZE_2));
        double xavierOut = Math.sqrt(6.0 / (Config.HIDDEN_SIZE_2 + Config.OUTPUT_SIZE));

        weightsInputHidden1 = new Matrix(Config.INPUT_SIZE, Config.HIDDEN_SIZE_1);
        weightsInputHidden1.randomize(-xavierL1, xavierL1);

        weightsHidden1Hidden2 = new Matrix(Config.HIDDEN_SIZE_1, Config.HIDDEN_SIZE_2);
        weightsHidden1Hidden2.randomize(-xavierL2, xavierL2);

        weightsHidden2Output = new Matrix(Config.HIDDEN_SIZE_2, Config.OUTPUT_SIZE);
        weightsHidden2Output.randomize(-xavierOut, xavierOut);

        biasHidden1 = new Matrix(1, Config.HIDDEN_SIZE_1);
        biasHidden2 = new Matrix(1, Config.HIDDEN_SIZE_2);
        biasOutput  = new Matrix(1, Config.OUTPUT_SIZE);
        // biases left as zero — standard with Xavier init
    }

    //returns output layer of neural network
    public Matrix forward(Matrix inputMatrix) {
        lastInput = inputMatrix;

        // hidden layer 1
        lastHidden1 = inputMatrix.mul(weightsInputHidden1);
        lastHidden1 = lastHidden1.elementAdd(biasHidden1);
        lastHidden1 = lastHidden1.map(x -> Math.tanh(x));      //squishes from -1 to 1

        // hidden layer 2
        lastHidden2 = lastHidden1.mul(weightsHidden1Hidden2);
        lastHidden2 = lastHidden2.elementAdd(biasHidden2);
        lastHidden2 = lastHidden2.map(x -> Math.tanh(x));      //squishes from -1 to 1

        // output layer
        Matrix outputLayer = lastHidden2.mul(weightsHidden2Output);
        outputLayer = outputLayer.elementAdd(biasOutput);
        outputLayer = softmax(outputLayer);   //makes probability vector, all elements sum up to 1

        return outputLayer;
    }

    //takes 6 raw numbers and convert them into 6 probabilities that add up to 1
    //does that through the e^x function because it always outputs a positive number no matter what
    //e^(-5)  =  0.0067
    //e^(0)   =  1.0
    //e^(5)   =  148.4
    private Matrix softmax(Matrix inputMatrix) {
        Matrix resultMatrix = new Matrix(1, Config.OUTPUT_SIZE);    //makes a new horizontal vector
        double sum = 0.0;

        for (int i = 0; i < Config.OUTPUT_SIZE; i++) {
            sum += Math.exp(inputMatrix.get(0, i));
        }

        for (int i = 0; i < Config.OUTPUT_SIZE; i++) {
            resultMatrix.set(0, i, Math.exp(inputMatrix.get(0, i)) / sum);      //divide each element by the sum
        }

        return resultMatrix;
    }

    //makes prediction on what emotion it is
    public int predict(Matrix inputMatrix) {
        Matrix emotionProbabilities = forward(inputMatrix);    //output of forward, each value is how confident the network is about that emotion

        int predictedLabel = 0;
        for (int i = 1; i < Config.OUTPUT_SIZE; i++) {
            if (emotionProbabilities.get(0, i) > emotionProbabilities.get(0, predictedLabel)) {
                predictedLabel = i;
            }
        }
        return predictedLabel;       //what the network thinks is the correct answer to later be compared with label parameter in backward
    }

    //goes from output to input (right to left), this is the backward propagation
    public void backward(Matrix resultMatrix, int label) {
        Matrix whichEmotion = new Matrix(1, Config.OUTPUT_SIZE);    //which emotion is correct vector
        whichEmotion.set(0, label, 1.0);                      //set 1 to the correct one

        Matrix outputError = resultMatrix.elementSubtract(whichEmotion);

        // gradients at weightsHidden2Output
        Matrix weightsHidden2Output2 = lastHidden2.transpose().mul(outputError);
        Matrix biasOutput2 = outputError;

        // propagate error back to hidden layer 2
        Matrix hidden2Error = outputError.mul(weightsHidden2Output.transpose());
        Matrix tanhDerivativeHidden2 = lastHidden2.map(h -> 1.0 - h * h);   //how steep was the tanh curve at this point
        Matrix deltaHidden2 = hidden2Error.elementMultiply(tanhDerivativeHidden2);

        // gradients at weightsHidden1Hidden2
        Matrix weightsHidden1Hidden22 = lastHidden1.transpose().mul(deltaHidden2);
        Matrix biasHidden22 = deltaHidden2;

        // propagate error back to hidden layer 1
        Matrix hidden1Error = deltaHidden2.mul(weightsHidden1Hidden2.transpose());
        Matrix tanhDerivativeHidden1 = lastHidden1.map(h -> 1.0 - h * h);   //how steep was the tanh curve at this point
        Matrix deltaHidden1 = hidden1Error.elementMultiply(tanhDerivativeHidden1);

        // gradients at weightsInputHidden1
        Matrix weightsInputHidden12 = lastInput.transpose().mul(deltaHidden1);
        Matrix biasHidden12 = deltaHidden1;

        //add learning rate so that it makes up for mistakes if we get something wrong
        weightsHidden2Output  = weightsHidden2Output.elementSubtract(weightsHidden2Output2.scale(Config.LEARNING_RATE));
        biasOutput            = biasOutput.elementSubtract(biasOutput2.scale(Config.LEARNING_RATE));
        weightsHidden1Hidden2 = weightsHidden1Hidden2.elementSubtract(weightsHidden1Hidden22.scale(Config.LEARNING_RATE));
        biasHidden2           = biasHidden2.elementSubtract(biasHidden22.scale(Config.LEARNING_RATE));
        weightsInputHidden1   = weightsInputHidden1.elementSubtract(weightsInputHidden12.scale(Config.LEARNING_RATE));
        biasHidden1           = biasHidden1.elementSubtract(biasHidden12.scale(Config.LEARNING_RATE));
    }
}
