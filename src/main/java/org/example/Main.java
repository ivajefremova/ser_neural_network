package org.example;

import audio.AudioLoader;
import audio.DatasetSplitter;
import audio.LabeledAudio;
import config.Config;
import config.Emotion;
import evaluation.Evaluator;
import features.FeatureNormalizer;
import features.FeatureVector;
import features.MFCCExtractor;
import math.Matrix;
import network.ModelSaver;
import network.NeuralNetwork;
import training.Training;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {

        // 1. load and split by actor
        System.out.println("Loading dataset from: " + Config.AUDIO_FOLDER);
        List<LabeledAudio> dataset = AudioLoader.loadDataset(Config.AUDIO_FOLDER);
        System.out.println("Loaded " + dataset.size() + " files.");

        List<List<LabeledAudio>> split = DatasetSplitter.split(dataset);
        List<LabeledAudio> trainAudio = split.get(0);
        List<LabeledAudio> testAudio  = split.get(1);
        System.out.println("Train: " + trainAudio.size() + "  |  Test: " + testAudio.size());

        // 2. extract MFCC features
        System.out.println("\nExtracting features...");
        List<FeatureVector> trainFeatures = new ArrayList<>();
        List<FeatureVector> testFeatures  = new ArrayList<>();

        for (LabeledAudio a : trainAudio) {
            double[] f = MFCCExtractor.extract(a.getFilePath());
            trainFeatures.add(new FeatureVector(f, a.getLabel()));
        }
        for (LabeledAudio a : testAudio) {
            double[] f = MFCCExtractor.extract(a.getFilePath());
            testFeatures.add(new FeatureVector(f, a.getLabel()));
        }

        // 3. normalize — fit only on training data, apply to both
        FeatureNormalizer normalizer = new FeatureNormalizer();
        normalizer.fit(trainFeatures);
        List<FeatureVector> trainNorm = normalizer.normalizeAll(trainFeatures);
        List<FeatureVector> testNorm  = normalizer.normalizeAll(testFeatures);

        // 4. train
        System.out.println("\nTraining for " + Config.EPOCHS + " epochs...");
        NeuralNetwork network = new NeuralNetwork();
        Training.train(network, trainNorm);

        // 5. save model
        ModelSaver.save(network, Config.MODEL_PATH);
        System.out.println("\nModel saved to " + Config.MODEL_PATH);

        // 6. evaluate on full test set
        System.out.println("\n--- Test Set Evaluation ---");
        Evaluator.evaluate(network, testNorm);

        // 7. single-file demo prediction
        // use args[0] if provided, otherwise pick the first file from the test set
        System.out.println("\n--- Single File Prediction ---");
        String demoPath;
        Emotion trueLabel = null;

        if (args.length > 0) {
            demoPath = args[0];
        } else {
            LabeledAudio demo = testAudio.get(0);
            demoPath = demo.getFilePath();
            trueLabel = demo.getLabel();
        }

        System.out.println("File: " + demoPath);

        double[] raw = MFCCExtractor.extract(demoPath);
        double[] normed = normalizer.normalize(raw);

        Matrix input = new Matrix(1, normed.length);
        for (int i = 0; i < normed.length; i++) input.set(0, i, normed[i]);

        Matrix probs = network.forward(input);

        int predicted = 0;
        for (int i = 1; i < Config.OUTPUT_SIZE; i++) {
            if (probs.get(0, i) > probs.get(0, predicted)) predicted = i;
        }

        System.out.println("\nConfidence scores:");
        for (Emotion e : Emotion.values()) {
            int i = e.ordinal();
            double pct = probs.get(0, i) * 100;
            String bar = "█".repeat((int)(pct / 5));
            System.out.printf("  %-10s %5.1f%%  %s%n", e.name(), pct, bar);
        }

        System.out.println("\nPredicted:  " + Emotion.findEmotion(predicted).name());
        if (trueLabel != null) {
            String correct = trueLabel.ordinal() == predicted ? "CORRECT" : "WRONG";
            System.out.println("True label: " + trueLabel.name() + "  →  " + correct);
        }
    }
}
