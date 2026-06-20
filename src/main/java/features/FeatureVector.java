package features;

import config.Emotion;

//vector that holds 31 numbers from the audio file that represent features and the emotion of that file
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
