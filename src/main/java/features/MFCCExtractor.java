package features;

import config.Config;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.File;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.apache.commons.math3.transform.FastCosineTransformer;
import org.apache.commons.math3.transform.DctNormalization;
import java.util.Arrays;

public class MFCCExtractor {
    private static final int FRAME_SIZE = 512;
    private static final int HOP_SIZE = 256;
    private static final int NUM_FILTERS = 33;
    private static final int NUM_COEFFS = 13;


    private static double hzToMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700.0);
    }

    private static double melToHz(double mel) {
        return 700 * (Math.pow(10, mel / 2595.0) - 1);
    }

    public static double[] readWAV(String filePath) throws Exception{
        File file = new File(filePath);
        AudioInputStream ais = AudioSystem.getAudioInputStream(file);

        byte[] buffer = ais.readAllBytes();

        double[] samples = new double[buffer.length / 2];

        for (int i = 0; i < buffer.length; i += 2) {
            int sample = (buffer[i+1] << 8) | (buffer[i] & 0xFF);
            samples[i/2] = sample / 32768.0;  //dividing scales it to [-1.0;1.0]
        }

        ais.close();
        return samples;
    }

    private static double[][] frameSignal(double[] samples){
        int numFrames = (samples.length - FRAME_SIZE) / HOP_SIZE + 1;
        double[][] frames = new double[numFrames][FRAME_SIZE];
        for (int i =0; i<numFrames; i++){

            int start = i * HOP_SIZE;
            System.arraycopy(samples, start, frames[i],0 , FRAME_SIZE);

        }
        return frames;
    }

    private static void applyHammingWindow(double[] frame){
        int N = frame.length;

        for(int n =0; n< N; n++){
            double window = 0.54 - 0.46 * Math.cos(Math.PI * n / (N-1));
            frame[n] = frame[n] * window;
        }
    }

    private static double[] computePowerSpectrum(double[] frame){
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] spectrum = fft.transform(frame, TransformType.FORWARD);
        double[] res = new double[FRAME_SIZE/2];
        for (int i = 0; i< FRAME_SIZE/2; i++){
            double real = spectrum[i].getReal();
            double imag = spectrum[i].getImaginary();
            double power = (real * real) + (imag * imag);
            res[i] = power;
        }
        return res;
    }

    private static double[] applyMelFilterbank(double[] powerSpectrum){


        int numFilters = NUM_FILTERS;
        double[] filterbank = new double[numFilters];

        double melMin = hzToMel(0);
        double melMax = hzToMel(Config.SAMPLE_RATE / 2.0);

        double[] melPoints = new double[numFilters + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + i * (melMax - melMin) / (numFilters + 1);
        }

        int[] bins = new int[numFilters + 2];
        for (int i = 0; i < melPoints.length; i++) {
            double hz = melToHz(melPoints[i]);
            bins[i] = (int)((FRAME_SIZE + 1) * hz / Config.SAMPLE_RATE);
        }

        for (int m = 1; m <= numFilters; m++) {
            for (int k = bins[m-1]; k < bins[m]; k++) {
                filterbank[m-1] += powerSpectrum[k] * (double)(k - bins[m-1]) / (bins[m] - bins[m-1]);
            }
            for (int k = bins[m]; k < bins[m+1]; k++) {
                filterbank[m-1] += powerSpectrum[k] * (double)(bins[m+1] - k) / (bins[m+1] - bins[m]);
            }
        }
        return filterbank;

    }

    private static double[] applyLog(double[] filterbank) {
        double[] result = new double[filterbank.length];
        for (int i = 0; i < filterbank.length; i++) {
            result[i] = Math.log(filterbank[i] + 1e-10);  // +1e-10 avoids log(0)
        }
        return result;
    }

    private static double[] applyDCT(double[] logFilterbank) {
        FastCosineTransformer dct = new FastCosineTransformer(DctNormalization.STANDARD_DCT_I);
        double[] result = dct.transform(logFilterbank, TransformType.FORWARD);

        return Arrays.copyOfRange(result, 0, NUM_COEFFS);
    }

    public static double[] extract(String filePath) throws Exception {
        double[] rwav = readWAV(filePath);
        double[][] frames = frameSignal(rwav);

        double[][] allMFCCs = new double[frames.length][NUM_COEFFS];
        for (int i = 0; i < frames.length; i++) {
            applyHammingWindow(frames[i]);
            double[] power = computePowerSpectrum(frames[i]);
            double[] filterbank = applyMelFilterbank(power);
            double[] logged = applyLog(filterbank);
            allMFCCs[i] = applyDCT(logged);
        }

        double[] result = new double[NUM_COEFFS * 2];
        for (int coeff = 0; coeff < NUM_COEFFS; coeff++) {

            // mean
            double sum = 0;
            for (int frame = 0; frame < allMFCCs.length; frame++) {
                sum += allMFCCs[frame][coeff];
            }
            double mean = sum / allMFCCs.length;
            result[coeff] = mean;

            // std
            double variance = 0;
            for (int frame = 0; frame < allMFCCs.length; frame++) {
                variance += Math.pow(allMFCCs[frame][coeff] - mean, 2);
            }
            result[NUM_COEFFS + coeff] = Math.sqrt(variance / allMFCCs.length);
        }
        return result;
    }


    public static void main(String[] args) throws Exception {
        String testFile = "/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV/1001_DFA_ANG_XX.wav";
        double[] vector = extract(testFile);
        System.out.println("\nFinal feature vector (" + vector.length + " values):");
        for (int i = 0; i < vector.length; i++) {
            System.out.println(i + ": " + vector[i]);
        }
    }

}
