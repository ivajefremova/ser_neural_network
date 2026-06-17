package config;

public class Config {
    public static final int INPUT_SIZE = 26;    // number of MFCC features
    public static final int HIDDEN_SIZE = 64;   // neurons in hidden layer
    public static final int OUTPUT_SIZE = 6;    // one per emotion

    // training settings
    public static final double LEARNING_RATE = 0.01;
    public static final int EPOCHS = 1000;      // how many times to loop over all training data

    // file paths
    public static final String MODEL_PATH = "src/main/resources/model.txt";
}
