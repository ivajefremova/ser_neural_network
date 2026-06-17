package audio;

//splits the data into two sets (training and testing) in 80:20 proportion, and returns those
//in a way that the same voice actor will never appear in both

import java.io.File;
import java.util.*;
//The * is a wildcard import — it imports everything from the java.util package at once instead of listing separately.
//  import java.util.List;
//  import java.util.ArrayList;
//  import java.util.Set;
//  import java.util.HashSet;

public class DatasetSplitter {
    public static List<List<LabeledAudio>> split(List<LabeledAudio> dataset){

        List<String> actorList = new ArrayList<>();     //list of strings of each actor
        for (LabeledAudio audio : dataset) {
            String actorId = new File(audio.getFilePath()).getName().split("_")[0];    //the numbers at the start are actor id
            if (!actorList.contains(actorId)) {
                actorList.add(actorId);
            }
        }
        Collections.shuffle(actorList);    //method from util* to mix the actor list

        int splitPoint = (int)(actorList.size() * 0.8);             //the split 80 for training 20 for testing

        Set<String> trainActors = new HashSet<>(actorList.subList(0, splitPoint));      //sublist and set are from list interface, sublist gets the sublist until the spliypoint
        Set<String> testActors  = new HashSet<>(actorList.subList(splitPoint, actorList.size()));

        List<LabeledAudio> trainingList = new ArrayList<>();
        List<LabeledAudio> testList = new ArrayList<>();

        //add some actors in training list and some in test list
        for (LabeledAudio a : dataset) {
            String actorId = new File(a.getFilePath()).getName().split("_")[0];
            if (trainActors.contains(actorId)) {
                trainingList.add(a);
            } else {
                testList.add(a);
            }
        }

        return Arrays.asList(trainingList, testList); //list of 2 sets training and test list
    }
}
