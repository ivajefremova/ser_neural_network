package features;

import java.util.ArrayList;
import java.util.List;

public class FeatureNormalizer {
    private double[] means;
    private double[] stds;

    public void fit(List<FeatureVector> trainingData){
        means = new double[26];
        stds = new double[26];
        int n = trainingData.size();

        for(int f =0; f < 26; f++){
            double sum = 0;
            for(FeatureVector ft : trainingData){
                sum += ft.getFeatures()[f];
            }
            means[f] = sum/n;
        }

        for(int f =0; f<26; f++){
            double var = 0;
            for(FeatureVector ft : trainingData){
                var += (ft.getFeatures()[f] - means[f])* (ft.getFeatures()[f] - means[f]);
            }
            stds[f] = Math.sqrt(var/n);
        }
    }

    public double[] normalize(double[] features){
        double[] normFeatures = new double[features.length];
        int count = 0;
        for(double val : features){
            double normVal = (val - means[count])/stds[count];
            normFeatures[count] = normVal;
            count += 1;
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
