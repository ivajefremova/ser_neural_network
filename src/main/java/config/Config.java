package config;

public class Config {

    // audio
    public static final int SAMPLE_RATE = 16000;
    public static final String AUDIO_FOLDER = "src/main/resources/data/AudioWAV";

    // network dimensions
    public static final int INPUT_SIZE = 31;
    public static final int HIDDEN_SIZE_1 = 64;
    public static final int HIDDEN_SIZE_2 = 32;
    public static final int OUTPUT_SIZE = 6;    // one per emotion

    // training settings
    public static final double LEARNING_RATE = 0.003;
    public static final int EPOCHS = 250;

    // data split
    public static final double TRAIN_SPLIT = 0.8;   // 80% training, 20% testing

    // file paths
    public static final String FEATURES_CSV = "features/features.csv";
    public static final String MODEL_PATH    = "models/model.txt";
    public static final String NORM_PATH     = "models/normalizer.txt";
}
