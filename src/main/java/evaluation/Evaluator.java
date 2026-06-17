package evaluation;

import config.Emotion;
import features.FeatureVector;
import math.Matrix;
import network.NeuralNetwork;

import java.util.List;

public class Evaluator {

    //loops through the test set, predicts each emotion, tracks correct/total overall and per emotion, prints the results
    public static void evaluate(NeuralNetwork nn, List<FeatureVector> testData) {
        int total = testData.size();
        int correct = 0;

        int[] correctPerEmotion = new int[Emotion.values().length];     //array of ints of the enum values
        int[] totalPerEmotion = new int[Emotion.values().length];      //same but will have diff purpose

        for (FeatureVector fv : testData) {
            Matrix input = toMatrix(fv.getFeatures());        //convert vector/list from audio into a matrix
            int predicted = nn.predict(input);        //predicted emotion
            int actual = fv.getLabel().ordinal();        //getLabel getter from feature vector gets actual emotion

            totalPerEmotion[actual]++;         //increments the value in the array at the index of that emotions enum
            if (predicted == actual) {
                correct++;
                correctPerEmotion[actual]++;      //increments the correctness of that emotion in the list if its predicted correctly
            }
        }

        System.out.printf("overall accuracy: %.1f%% (%d/%d)%n", 100.0 * correct / total, correct, total);
        System.out.println();

        for (Emotion e : Emotion.values()) {
            int i = e.ordinal();
            double acc = totalPerEmotion[i] == 0 ? 0 : 100.0 * correctPerEmotion[i] / totalPerEmotion[i];
            System.out.printf("  %-10s: %.1f%%  (%d/%d)%n", e.name(), acc, correctPerEmotion[i], totalPerEmotion[i]);  //%% means literal % sign
        }                                               //%d whole number (integer), %n is neewline
    }

    //private helper that converts double[] to a Matrix [1×26] network
    private static Matrix toMatrix(double[] features) {
        Matrix m = new Matrix(1, features.length);
        for (int j = 0; j < features.length; j++) {
            m.set(0, j, features[j]);
        }
        return m;
    }
}
