package network;

import config.Config;
import math.Matrix;

// inputMatrix        [1 × 40]   — one row, 40 MFCC features
//  weightsHiddenInput [40 × 64]  — connects every input to every hidden neuron
//  biasHidden         [1 × 64]   — one bias per hidden neuron
//  weightsHiddenOutput[64 × 6]   — connects every hidden neuron to every emotion
//  biasOutput         [1 × 6]    — one bias per emotion output

public class NeuralNetwork {
    private Matrix weightsHiddenInput;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasOutput;
    private Matrix lastInput;
    private Matrix lastHidden;

    //getters and setters for model saver
    public Matrix getWeightsHiddenInput()  { return weightsHiddenInput; }
    public Matrix getWeightsHiddenOutput() { return weightsHiddenOutput; }
    public Matrix getBiasHidden()          { return biasHidden; }
    public Matrix getBiasOutput()          { return biasOutput; }

    public void setWeightsHiddenInput(Matrix m)  { weightsHiddenInput = m; }
    public void setWeightsHiddenOutput(Matrix m) { weightsHiddenOutput = m; }
    public void setBiasHidden(Matrix m)          { biasHidden = m; }
    public void setBiasOutput(Matrix m)          { biasOutput = m; }

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
        this.lastInput = inputMatrix;

        // hidden layer
        Matrix hiddenLayer = inputMatrix.mul(weightsHiddenInput);  //internal layer is gotten by multiplying input and hidden input matrix
        hiddenLayer = hiddenLayer.elementAdd(biasHidden);      //add the bias for calculation
        hiddenLayer = hiddenLayer.map(x -> Math.tanh(x));      //squishes from -1 to 1

        this.lastHidden = hiddenLayer;

        // output layer
        Matrix outputLayer = hiddenLayer.mul(weightsHiddenOutput);   //get the output layer by matrix multiplication of hiddenlayer with hidden output
        outputLayer = outputLayer.elementAdd(biasOutput);  //add the bias to the output
        outputLayer = softmax(outputLayer);   //makes probability vector, all elements sum up to 1

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

        Matrix errorMatrix = resultMatrix.elementSubtract(whichEmotion);

        Matrix weightsHiddenOutput2 = lastHidden.transpose().mul(errorMatrix);  //same as weightsHiddenOutput (its gradient)
        Matrix biasOutput2 = errorMatrix;                                 //same as biasOurput (its gradient), it's just the error no multiplication

        Matrix layer1Error = errorMatrix.mul(weightsHiddenOutput.transpose());      //back once more

        Matrix tanhDerivative = lastHidden.map(h -> 1.0 - h * h);       //how steep was the tanh curve at this point? to go even further back
        Matrix beforeTanh = layer1Error.elementMultiply(tanhDerivative);                 //rom squshed to initial

        Matrix weightsHiddenInput2 = lastInput.transpose().mul(beforeTanh);   //final multiply layer (back to the first one)
        Matrix biasHidden2 = beforeTanh;

        //add learning rate so that it makes up for mistakes if we get something wrong
        weightsHiddenOutput = weightsHiddenOutput.elementSubtract(weightsHiddenOutput2.scale(Config.LEARNING_RATE));
        biasOutput = biasOutput.elementSubtract(biasOutput2.scale(Config.LEARNING_RATE));
        weightsHiddenInput  = weightsHiddenInput.elementSubtract(weightsHiddenInput2.scale(Config.LEARNING_RATE));
        biasHidden = biasHidden.elementSubtract(biasHidden2.scale(Config.LEARNING_RATE));
    }

}
