import config.Config;
import config.Emotion;
import features.FeatureNormalizer;
import features.MFCCExtractor;
import math.Matrix;
import network.ModelSaver;
import network.NeuralNetwork;

import javax.sound.sampled.*;
import java.io.File;
import java.util.concurrent.CountDownLatch;

public class Predict {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: Predict <path-to-wav-file>");
            System.exit(1);
        }

        String wavPath = args[0];
        File wavFile = new File(wavPath);
        if (!wavFile.exists()) {
            System.out.println("File not found: " + wavPath);
            System.exit(1);
        }

        // load model and normalizer
        System.out.println("Loading model...");
        NeuralNetwork network = ModelSaver.load(Config.MODEL_PATH);
        FeatureNormalizer normalizer = FeatureNormalizer.load(Config.NORM_PATH);

        // extract and normalize features
        System.out.println("Extracting features from: " + wavFile.getName());
        double[] raw = MFCCExtractor.extract(wavPath);
        double[] normalized = normalizer.normalize(raw);

        Matrix input = new Matrix(1, normalized.length);
        for (int i = 0; i < normalized.length; i++) input.set(0, i, normalized[i]);

        // run prediction
        Matrix output = network.forward(input);

        int bestIdx = 0;
        for (int i = 1; i < Config.OUTPUT_SIZE; i++) {
            if (output.get(0, i) > output.get(0, bestIdx)) bestIdx = i;
        }
        Emotion predicted = Emotion.values()[bestIdx];
        double confidence = output.get(0, bestIdx) * 100;

        // print all probabilities
        System.out.println("\n--- Emotion Probabilities ---");
        Emotion[] emotions = Emotion.values();
        for (int i = 0; i < Config.OUTPUT_SIZE; i++) {
            String marker = i == bestIdx ? " <--" : "";
            System.out.printf("  %-10s %.1f%%%s%n", emotions[i], output.get(0, i) * 100, marker);
        }
        System.out.printf("%nPrediction: %s (%.1f%% confidence)%n", predicted, confidence);

        // play the audio file
        System.out.println("\nPlaying audio...");
        playWav(wavPath);

        // announce result via text-to-speech
        String announcement = "I detected the emotion: " + predicted.toString().toLowerCase();
        speak(announcement);
    }

    private static void playWav(String path) {
        try {
            AudioInputStream ais = AudioSystem.getAudioInputStream(new File(path));
            Clip clip = AudioSystem.getClip();
            clip.open(ais);

            CountDownLatch latch = new CountDownLatch(1);
            clip.addLineListener(e -> {
                if (e.getType() == LineEvent.Type.STOP) latch.countDown();
            });

            clip.start();
            latch.await();
            clip.close();
        } catch (Exception e) {
            System.err.println("Audio playback failed: " + e.getMessage());
        }
    }

    private static void speak(String text) {
        try {
            new ProcessBuilder("say", text).inheritIO().start().waitFor();
        } catch (Exception e) {
            System.err.println("Text-to-speech failed: " + e.getMessage());
        }
    }
}
