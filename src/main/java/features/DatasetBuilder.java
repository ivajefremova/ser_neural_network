package features;

import audio.AudioLoader;
import audio.LabeledAudio;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import static features.MFCCExtractor.extract;

public class DatasetBuilder {
    public static void saveToCSV(List<FeatureVector> vectors, String outputPath) throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));

        for (FeatureVector fv : vectors) {
            StringBuilder row = new StringBuilder();
            row.append(fv.getLabel().name());
            for (double val : fv.getFeatures()) {
                row.append(",").append(val);
            }
            writer.write(row.toString());
            writer.newLine();
        }
        writer.close();
        System.out.println("Saved");
    }

    public static void dataSave(String path) throws Exception{
        List<LabeledAudio> dataset = AudioLoader.loadDataset(path);
        List<FeatureVector> vectors = new ArrayList<>();
        int count = 0;
        for (LabeledAudio audio : dataset){
            double[] ft = extract(audio.getFilePath());
            vectors.add(new FeatureVector(ft, audio.getLabel()));
            count++;
            if (count % 500 == 0) {
                System.out.println("Processed " + count + "/" + dataset.size());
            }
        }
        saveToCSV(vectors, "/Users/oskar/Documents/NeuroNetProject/ser_neural_network/src/main/java/features/features.csv");
    }
    public static void main(String[] args) throws Exception{
        dataSave("/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV");
    }
}
