package audio;

import config.Emotion;

//identifies an emotion from the file's name

public class LabelParser {
    public static Emotion parse(String filename){
        String[] myArray = filename.split("_");
        String emotionCode = myArray[2];
        switch(emotionCode) {
            case "ANG": return Emotion.ANGRY;
            case "DIS": return Emotion.DISGUST;
            case "FEA": return Emotion.FEARFUL;
            case "HAP": return Emotion.HAPPY;
            case "NEU": return Emotion.NEUTRAL;
            case "SAD": return Emotion.SAD;
            default: return null; // unknown code
        }
    }

    //just tests if it works
    public static void main(String[] args){
        System.out.println(parse("1001_DFA_ANG_XX.wav"));
        System.out.println(parse("1002_IEO_HAP_HI.wav"));
        System.out.println(parse("1003_IEO_SAD_LO.wav"));
    }
}
