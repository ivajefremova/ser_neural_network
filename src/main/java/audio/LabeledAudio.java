package audio;
import config.Emotion;

//Creates an object that stores an emotion and the path to the audio file

public class LabeledAudio {
    private String filePath;
    private Emotion label;

    public LabeledAudio(String filePath, Emotion label){
        this.filePath = filePath;
        this.label = label;
    }

    public String getFilePath(){
        return filePath;
    }

    public Emotion getLabel(){
        return label;
    }
}
