import audio.AudioLoader;
import audio.DatasetSplitter;
import audio.LabeledAudio;
import config.Config;
import config.Emotion;
import evaluation.Evaluator;
import features.FeatureNormalizer;
import features.FeatureVector;
import features.MFCCExtractor;
import network.ModelSaver;
import network.NeuralNetwork;
import math.Matrix;
import training.Training;

import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {

        // 1 - load and split
        System.out.println("Loading dataset...");
        List<LabeledAudio> dataset = AudioLoader.loadDataset(Config.AUDIO_FOLDER);
        List<List<LabeledAudio>> split = DatasetSplitter.split(dataset);
        List<LabeledAudio> trainAudio = split.get(0);
        List<LabeledAudio> testAudio  = split.get(1);

        // 2 - extract features
        System.out.println("Extracting features...");
        List<FeatureVector> trainFeatures = new ArrayList<>();
        List<FeatureVector> testFeatures  = new ArrayList<>();

        for (LabeledAudio a : trainAudio) {
            trainFeatures.add(new FeatureVector(MFCCExtractor.extract(a.getFilePath()), a.getLabel()));
        }
        for (LabeledAudio a : testAudio) {
            testFeatures.add(new FeatureVector(MFCCExtractor.extract(a.getFilePath()), a.getLabel()));
        }

        // 3 - normalize
        FeatureNormalizer normalizer = new FeatureNormalizer();
        normalizer.fit(trainFeatures);
        List<FeatureVector> trainNorm = normalizer.normalizeAll(trainFeatures);
        List<FeatureVector> testNorm  = normalizer.normalizeAll(testFeatures);

        // 4 - pick demo samples (one per emotion)
        List<FeatureVector> demoSamples = pickDemoSamples(testNorm);

        // 5 - create network
        NeuralNetwork network = new NeuralNetwork();

        // 6 - BEFORE training
        System.out.println("\n=== BEFORE TRAINING (random weights) ===");
        showPredictions(network, demoSamples);

        // 7 - train
        System.out.println("\n=== TRAINING ===");
        Training.train(network, trainNorm, testNorm);

        // 8 - AFTER training, same samples
        System.out.println("\n=== AFTER TRAINING (same samples) ===");
        showPredictions(network, demoSamples);

        // 9 - full test set evaluation
        System.out.println("\n=== FULL TEST SET EVALUATION ===");
        Evaluator.evaluate(network, testNorm);

        // 10 - save model and normalizer
        ModelSaver.save(network, Config.MODEL_PATH);
        normalizer.save(Config.NORM_PATH);
        System.out.println("\nModel saved to " + Config.MODEL_PATH);
        System.out.println("Normalizer saved to " + Config.NORM_PATH);
    }

    private static List<FeatureVector> pickDemoSamples(List<FeatureVector> testData) {
        List<FeatureVector> demo = new ArrayList<>();
        Set<Emotion> seen = new HashSet<>();
        for (FeatureVector fv : testData) {
            if (!seen.contains(fv.getLabel())) {
                demo.add(fv);
                seen.add(fv.getLabel());
            }
            if (seen.size() == Emotion.values().length) break;
        }
        return demo;
    }

    private static void showPredictions(NeuralNetwork network, List<FeatureVector> samples) {
        for (FeatureVector fv : samples) {
            Matrix input = toMatrix(fv.getFeatures());
            Matrix output = network.forward(input);

            int predictedIndex = argmax(output);
            Emotion predicted = Emotion.values()[predictedIndex];
            Emotion actual = fv.getLabel();
            double confidence = output.get(0, predictedIndex) * 100;

            String mark = predicted == actual ? "correct" : "wrong";
            System.out.printf("Actual: %-10s Predicted: %-10s (%.1f%% confidence) — %s%n",
                    actual, predicted, confidence, mark);
        }
    }

    private static int argmax(Matrix output) {
        int best = 0;
        for (int i = 1; i < output.getCols(); i++) {
            if (output.get(0, i) > output.get(0, best)) best = i;
        }
        return best;
    }

    private static Matrix toMatrix(double[] features) {
        Matrix m = new Matrix(1, features.length);
        for (int i = 0; i < features.length; i++) {
            m.set(0, i, features[i]);
        }
        return m;
    }
}