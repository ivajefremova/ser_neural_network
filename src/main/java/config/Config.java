package config;

public class Config {

    // audio
    public static final int SAMPLE_RATE = 16000;
    public static final String AUDIO_FOLDER = "src/main/resources/data/AudioWAV";

    // network dimensions
    public static final int INPUT_SIZE = 26;    // 13 MFCC means + 13 MFCC stds
    public static final int HIDDEN_SIZE = 64;   // neurons in hidden layer
    public static final int OUTPUT_SIZE = 6;    // one per emotion

    // training settings
    public static final double LEARNING_RATE = 0.01;
    public static final int EPOCHS = 100;

    // data split
    public static final double TRAIN_SPLIT = 0.8;   // 80% training, 20% testing

    // file paths
    public static final String FEATURES_CSV = "features/features.csv";
    public static final String MODEL_PATH = "src/main/resources/model.txt";
}
