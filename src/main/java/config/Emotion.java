package config;

public enum Emotion {
    HAPPY,
    SAD,
    ANGRY,
    DISGUST,
    FEARFUL,
    NEUTRAL;

    public String getLabel() {
        return this.name();
    }
    //returns the index
    public int findIndex() {
        return this.ordinal();
    }

    //takes index and finds its associated emotion
    public static Emotion findEmotion(int index) {
        Emotion[] values = Emotion.values();
        if (index < 0 || index >= values.length) {
            throw new IllegalArgumentException("invalid emotion index: " + index +". only 0 to 5 allowed");
        }
        return values[index];
    }

}
