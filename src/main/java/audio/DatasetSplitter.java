package audio;

//Splits the data into two sets (training and testing) in 80:20 propotion, in a way that the same voice actor will never appear in both

import config.Config;
import java.io.File;
import java.util.*;

public class DatasetSplitter {
    public static List<List<LabeledAudio>> split(List<LabeledAudio> dataset){

        List<String> actorList = new ArrayList<>();
        for (LabeledAudio audio : dataset) {
            String actorId = new File(audio.getFilePath()).getName().split("_")[0];
            if (!actorList.contains(actorId)) {
                actorList.add(actorId);
            }
        }
        Collections.shuffle(actorList);

        int splitPoint = (int)(actorList.size() * Config.TRAIN_SPLIT);

        Set<String> trainActors = new HashSet<>(actorList.subList(0, splitPoint));
        Set<String> testActors  = new HashSet<>(actorList.subList(splitPoint, actorList.size()));

        List<LabeledAudio> trainingList = new ArrayList<>();
        List<LabeledAudio> testList = new ArrayList<>();

        for (LabeledAudio a : dataset) {
            String actorId = new File(a.getFilePath()).getName().split("_")[0];
            if (trainActors.contains(actorId)) {
                trainingList.add(a);
            } else {
                testList.add(a);
            }
        }

        return Arrays.asList(trainingList, testList);
    }
}
