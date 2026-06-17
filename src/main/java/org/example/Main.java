package org.example;

import audio.AudioLoader;
import audio.DatasetSplitter;
import audio.LabeledAudio;
import features.FeatureNormalizer;
import features.FeatureVector;
import features.MFCCExtractor;

import java.util.ArrayList;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        String dataPath = "/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV";

        // 1. load and split
        List<LabeledAudio> dataset = AudioLoader.loadDataset(dataPath);
        List<List<LabeledAudio>> split = DatasetSplitter.split(dataset);
        List<LabeledAudio> trainAudio = split.get(0);
        List<LabeledAudio> testAudio  = split.get(1);

        // 2. extract features
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

        // 3. normalize — fit on training only
        FeatureNormalizer normalizer = new FeatureNormalizer();
        normalizer.fit(trainFeatures);
        List<FeatureVector> trainNorm = normalizer.normalizeAll(trainFeatures);
        List<FeatureVector> testNorm  = normalizer.normalizeAll(testFeatures);

        // 4. verify
        System.out.println("Training samples: " + trainNorm.size());
        System.out.println("Test samples: "     + testNorm.size());
        System.out.println("\nFirst training vector (normalized):");
        for (double v : trainNorm.get(0).getFeatures()) {
            System.out.printf("%.4f  ", v);
        }
        System.out.println("\nLabel: " + trainNorm.get(0).getLabel());
    }
}
