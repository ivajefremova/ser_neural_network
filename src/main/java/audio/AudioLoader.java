package audio;

import config.Config;
import config.Emotion;
import java.io.File;     //class that represents directory
import java.util.List;    //interface for list , which has class ArrayList that implements all the methods
import java.util.ArrayList;   //class arraylist

//creates a list of objects LabeledAudio for every audio file in the folder
//mostly used File class from java

public class AudioLoader {
    public static List<LabeledAudio> loadDataset(String folderPath){
        File folder = new File(folderPath);
        File[] files = folder.listFiles();          //array of files in the folder

        ArrayList<LabeledAudio> listLabAudio = new ArrayList<LabeledAudio>();      //list of labeled audio objects, generic <>
        if (files == null) {
            System.out.println("Folder not found or empty: " + folderPath);
            return listLabAudio;
        }

        for (File f : files){
            if (f.getName().endsWith(".wav")){
                String name = f.getName();
                String absPath = f.getAbsolutePath();
                Emotion emo = LabelParser.parse(name);     //from labelParser gets the emotion from the filepath name
                LabeledAudio labAudio = new LabeledAudio(absPath, emo);
                listLabAudio.add(labAudio);     //method from arraylist , list interface
            }
        }
        return listLabAudio;      //list of this - ("/some/path/1001_DFA_ANG_XX.wav", ANGRY).
    }

    //prints the first 5 entries to test it
    public static void main(String[] args) {
        List<LabeledAudio> dataset = loadDataset(Config.AUDIO_FOLDER);
        System.out.println("Total files loaded: " + dataset.size());

        for (int i = 0; i < 5; i++) {
            System.out.println(dataset.get(i).getFilePath() + " → " + dataset.get(i).getLabel());
        }
    }

}
