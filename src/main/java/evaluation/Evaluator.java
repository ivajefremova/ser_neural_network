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

        int[] correctPerEmotion = new int[Emotion.values().length];
        int[] totalPerEmotion = new int[Emotion.values().length];

        for (FeatureVector fv : testData) {
            Matrix input = toMatrix(fv.getFeatures());
            int predicted = nn.predict(input);
            int actual = fv.getLabel().ordinal();

            totalPerEmotion[actual]++;
            if (predicted == actual) {
                correct++;
                correctPerEmotion[actual]++;
            }
        }

        System.out.printf("overall accuracy: %.1f%% (%d/%d)%n", 100.0 * correct / total, correct, total);
        System.out.println();

        for (Emotion e : Emotion.values()) {
            int i = e.ordinal();
            double acc = totalPerEmotion[i] == 0 ? 0 : 100.0 * correctPerEmotion[i] / totalPerEmotion[i];
            System.out.printf("  %-10s: %.1f%%  (%d/%d)%n", e.name(), acc, correctPerEmotion[i], totalPerEmotion[i]);
        }
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
