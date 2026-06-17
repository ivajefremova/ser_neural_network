package features;

import config.Config;
import java.util.ArrayList;
import java.util.List;

public class FeatureNormalizer {
    private double[] means;
    private double[] stds;

    public void fit(List<FeatureVector> trainingData){
        means = new double[Config.INPUT_SIZE];
        stds = new double[Config.INPUT_SIZE];
        int n = trainingData.size();

        for(int f = 0; f < Config.INPUT_SIZE; f++){
            double sum = 0;
            for(FeatureVector ft : trainingData){
                sum += ft.getFeatures()[f];
            }
            means[f] = sum / n;
        }

        for(int f = 0; f < Config.INPUT_SIZE; f++){
            double var = 0;
            for(FeatureVector ft : trainingData){
                var += (ft.getFeatures()[f] - means[f]) * (ft.getFeatures()[f] - means[f]);
            }
            stds[f] = Math.sqrt(var / n);
        }
    }

    public double[] normalize(double[] features){
        double[] normFeatures = new double[features.length];
        for(int i = 0; i < features.length; i++){
            normFeatures[i] = stds[i] == 0 ? 0 : (features[i] - means[i]) / stds[i];
        }
        return normFeatures;
    }

    public List<FeatureVector> normalizeAll(List<FeatureVector> data) {
        List<FeatureVector> result = new ArrayList<>();
        for (FeatureVector fv : data) {
            double[] normed = normalize(fv.getFeatures());
            result.add(new FeatureVector(normed, fv.getLabel()));
        }
        return result;
    }

}
