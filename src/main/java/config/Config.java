package config;

public class Config {


    public static final int SAMPLE_RATE = 16000;


    public static final String AUDIO_FOLDER = "/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV";
    public static final String FEATURES_CSV = "features/features.csv";


    public static final int INPUT_SIZE = 26;   // 13 MFCC means + 13 MFCC stds
    public static final int HIDDEN_SIZE = 64;  // neurons in the hidden layer
    public static final int OUTPUT_SIZE = 6;   // number of emotions


    public static final double LEARNING_RATE = 0.01;
    public static final int EPOCHS = 100;

}