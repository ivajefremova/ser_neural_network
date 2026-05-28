package config;

public class Config {
    public static final int INPUT_SIZE = 40;    // number of MFCC features
    public static final int HIDDEN_SIZE = 64;   // neurons in hidden layer
    public static final int OUTPUT_SIZE = 6;    // one per emotion

    // training settings
    public static final double LEARNING_RATE = 0.01;
    public static final int EPOCHS = 1000;      // how many times to loop over all training data

    // data split
    public static final double TRAIN_SPLIT = 0.8;   // 80% training, 20% testing

    // file paths
    public static final String DATA_PATH = "src/main/resources/data/";
    public static final String MODEL_PATH = "src/main/resources/model.txt";
}
