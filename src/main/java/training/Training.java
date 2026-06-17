package training;

import features.FeatureVector;
import math.Matrix;
import config.Config;
import network.NeuralNetwork;

import java.util.Collections;
import java.util.List;

public class Training {

    public static void train(NeuralNetwork network, List<FeatureVector> trainingData, List<FeatureVector> testData) {

        double bestTestAccuracy = 0;
        Matrix bestW1 = null, bestW2 = null, bestW3 = null;
        Matrix bestB1 = null, bestB2 = null, bestB3 = null;

        for (int epoch = 0; epoch < Config.EPOCHS; epoch++) {

            Collections.shuffle(trainingData);

            int correct = 0;
            double totalLoss = 0;

            for (FeatureVector fv : trainingData) {
                Matrix input = toMatrix(fv.getFeatures());
                int label = fv.getLabel().ordinal();

                Matrix output = network.forward(input);

                int predicted = 0;
                for (int i = 0; i < Config.OUTPUT_SIZE; i++) {
                    if (output.get(0, i) > output.get(0, predicted)) predicted = i;
                }
                if (predicted == label) correct++;

                double prob = output.get(0, label);
                totalLoss += -Math.log(prob + 1e-10);

                network.backward(output, label);
            }

            double trainAccuracy = (double) correct / trainingData.size();
            double avgLoss = totalLoss / trainingData.size();

            // check test accuracy every 5 epochs
            if (epoch % 1 == 0 || epoch == Config.EPOCHS - 1) {
                double testAccuracy = quickEvaluate(network, testData);

                System.out.printf("Epoch %d | Loss: %.4f | Train: %.2f%% | Test: %.2f%%%n",
                        epoch, avgLoss, trainAccuracy * 100, testAccuracy * 100);

                if (testAccuracy > bestTestAccuracy) {
                    bestTestAccuracy = testAccuracy;
                    bestW1 = network.getWeightsInputHidden1().copy();
                    bestW2 = network.getWeightsHidden1Hidden2().copy();
                    bestW3 = network.getWeightsHidden2Output().copy();
                    bestB1 = network.getBiasHidden1().copy();
                    bestB2 = network.getBiasHidden2().copy();
                    bestB3 = network.getBiasOutput().copy();
                }
            } else {
                System.out.printf("Epoch %d | Loss: %.4f | Train: %.2f%%%n",
                        epoch, avgLoss, trainAccuracy * 100);
            }
        }

        // restore the best version seen during training
        if (bestW1 != null) {
            network.setWeightsInputHidden1(bestW1);
            network.setWeightsHidden1Hidden2(bestW2);
            network.setWeightsHidden2Output(bestW3);
            network.setBiasHidden1(bestB1);
            network.setBiasHidden2(bestB2);
            network.setBiasOutput(bestB3);
            System.out.printf("%nRestored best model — test accuracy was %.2f%%%n", bestTestAccuracy * 100);
        }
    }

    private static double quickEvaluate(NeuralNetwork network, List<FeatureVector> testData) {
        int correct = 0;
        for (FeatureVector fv : testData) {
            Matrix input = toMatrix(fv.getFeatures());
            int predicted = network.predict(input);
            if (predicted == fv.getLabel().ordinal()) correct++;
        }
        return (double) correct / testData.size();
    }

    private static Matrix toMatrix(double[] vector) {
        Matrix mat = new Matrix(1, vector.length);
        for (int i = 0; i < vector.length; i++) {
            mat.set(0, i, vector[i]);
        }
        return mat;
    }
}