package features;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.File;

public class MFCCExtractor {
    public static double readWAV(String filePath) throws Exception{
        File file = new File(filePath);
        AudioInputStream ais = AudioSystem.getAudioInputStream(file);

        byte[] buffer = ais.readAllBytes();

        double[] samples = new double[buffer.length / 2];

        for (int i = 0; i < buffer.length; i += 2) {
            int sample = (buffer[i+1] << 8) | (buffer[i] & 0xFF);
            samples[i/2] = sample / 32768.0;  //dividing scales it to [-1.0;1.0]
        }
    }
}
