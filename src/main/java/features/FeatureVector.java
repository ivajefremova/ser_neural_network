package features;

import config.Emotion;

public class FeatureVector {
    private double[] features;
    private Emotion emo;

    public FeatureVector(double[] features, Emotion emo){
        this.features = features;
        this.emo = emo;
    }
    public double[] getFeatures(){
        return features;
    }

    public Emotion getLabel(){
        return emo;
    }
}
