package audio;

import config.Emotion;
import java.io.File;
import java.util.List;
import java.util.ArrayList;

//Creates a list of objects LabeledAudio for every audio file in the folder

public class AudioLoader {
    public static List<LabeledAudio> loadDataset(String folderPath){
        File folder = new File(folderPath);
        File[] files = folder.listFiles();
        ArrayList<LabeledAudio> listLabAudio = new ArrayList<LabeledAudio>();
        if (files == null) {
            System.out.println("Folder not found or empty: " + folderPath);
            return listLabAudio;
        }

        for (File f : files){
            if (f.getName().endsWith(".wav")){
                String name = f.getName();
                String absPath = f.getAbsolutePath();
                Emotion emo = LabelParser.parse(name);
                LabeledAudio labAudio = new LabeledAudio(absPath, emo);
                listLabAudio.add(labAudio);
            }
        }
        return listLabAudio;
    }

    public static void main(String[] args) {
        List<LabeledAudio> dataset = loadDataset("/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV");
        System.out.println("Total files loaded: " + dataset.size());

        for (int i = 0; i < 5; i++) {
            System.out.println(dataset.get(i).getFilePath() + " → " + dataset.get(i).getLabel());
        }
    }

}
