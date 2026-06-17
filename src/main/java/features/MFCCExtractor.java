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

    private static final int SAMPLE_RATE = 16000;
    private static final int FRAME_SIZE = 512;
    private static final int HOP_SIZE = 256;
    private static final int NUM_FILTERS = 33;
    private static final int NUM_COEFFS = 13;

    private static final int MIN_PITCH_HZ = 80;
    private static final int MAX_PITCH_HZ = 400;

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
            samples[i/2] = sample / 32768.0;
        }

        ais.close();
        return samples;
    }

    private static double[][] frameSignal(double[] samples){
        int numFrames = (samples.length - FRAME_SIZE) / HOP_SIZE + 1;
        double[][] frames = new double[numFrames][FRAME_SIZE];
        for (int i = 0; i < numFrames; i++){
            int start = i * HOP_SIZE;
            System.arraycopy(samples, start, frames[i], 0, FRAME_SIZE);
        }
        return frames;
    }

    private static void applyHammingWindow(double[] frame){
        int N = frame.length;
        for (int n = 0; n < N; n++){
            double window = 0.54 - 0.46 * Math.cos(2 * Math.PI * n / (N - 1)); // fixed
            frame[n] = frame[n] * window;
        }
    }

    private static double[] computePowerSpectrum(double[] frame){
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] spectrum = fft.transform(frame, TransformType.FORWARD);
        double[] res = new double[FRAME_SIZE/2];
        for (int i = 0; i < FRAME_SIZE/2; i++){
            double real = spectrum[i].getReal();
            double imag = spectrum[i].getImaginary();
            res[i] = (real * real) + (imag * imag);
        }
        return res;
    }

    private static double[] applyMelFilterbank(double[] powerSpectrum){
        int numFilters = NUM_FILTERS;
        double[] filterbank = new double[numFilters];

        double melMin = hzToMel(0);
        double melMax = hzToMel(SAMPLE_RATE / 2.0);

        double[] melPoints = new double[numFilters + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + i * (melMax - melMin) / (numFilters + 1);
        }

        int[] bins = new int[numFilters + 2];
        for (int i = 0; i < melPoints.length; i++) {
            double hz = melToHz(melPoints[i]);
            bins[i] = (int)((FRAME_SIZE + 1) * hz / SAMPLE_RATE);
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
            result[i] = Math.log(filterbank[i] + 1e-10);
        }
        return result;
    }

    private static double[] applyDCT(double[] logFilterbank) {
        FastCosineTransformer dct = new FastCosineTransformer(DctNormalization.STANDARD_DCT_I);
        double[] result = dct.transform(logFilterbank, TransformType.FORWARD);
        return Arrays.copyOfRange(result, 0, NUM_COEFFS);
    }

    // ===== NEW: energy of one frame =====
    private static double computeFrameEnergy(double[] frame) {
        double sum = 0;
        for (double v : frame) {
            sum += v * v;
        }
        return sum / frame.length;
    }

    // ===== NEW: pitch estimate of one frame via autocorrelation =====
    private static double estimatePitch(double[] frame) {
        int minLag = SAMPLE_RATE / MAX_PITCH_HZ;
        int maxLag = SAMPLE_RATE / MIN_PITCH_HZ;

        double bestCorrelation = -1;
        int bestLag = minLag;

        for (int lag = minLag; lag <= maxLag && lag < frame.length; lag++) {
            double correlation = 0;
            for (int n = 0; n < frame.length - lag; n++) {
                correlation += frame[n] * frame[n + lag];
            }
            if (correlation > bestCorrelation) {
                bestCorrelation = correlation;
                bestLag = lag;
            }
        }
        return (double) SAMPLE_RATE / bestLag;
    }

    // ===== NEW: zero crossing rate over the whole file =====
    private static double computeZCR(double[] samples) {
        int crossings = 0;
        for (int i = 1; i < samples.length; i++) {
            boolean signChanged = (samples[i] >= 0 && samples[i-1] < 0) ||
                    (samples[i] < 0 && samples[i-1] >= 0);
            if (signChanged) crossings++;
        }
        return (double) crossings / samples.length;
    }

    private static double average(double[] values) {
        double sum = 0;
        for (double v : values) sum += v;
        return sum / values.length;
    }

    private static double stdDev(double[] values, double mean) {
        double variance = 0;
        for (double v : values) variance += Math.pow(v - mean, 2);
        return Math.sqrt(variance / values.length);
    }

    public static double[] extract(String filePath) throws Exception {
        double[] rwav = readWAV(filePath);
        double[][] frames = frameSignal(rwav);

        double[][] allMFCCs = new double[frames.length][NUM_COEFFS];
        double[] frameEnergies = new double[frames.length];
        double[] framePitches = new double[frames.length];

        for (int i = 0; i < frames.length; i++) {
            // measure on the RAW frame before windowing changes it
            frameEnergies[i] = computeFrameEnergy(frames[i]);
            framePitches[i] = estimatePitch(frames[i]);

            applyHammingWindow(frames[i]);
            double[] power = computePowerSpectrum(frames[i]);
            double[] filterbank = applyMelFilterbank(power);
            double[] logged = applyLog(filterbank);
            allMFCCs[i] = applyDCT(logged);
        }

        double zcr = computeZCR(rwav);

        double[] result = new double[Config.INPUT_SIZE]; // 31

        // MFCC mean/std (26 values)
        for (int coeff = 0; coeff < NUM_COEFFS; coeff++) {
            double sum = 0;
            for (double[] mfccs : allMFCCs) sum += mfccs[coeff];
            double mean = sum / allMFCCs.length;
            result[coeff] = mean;

            double variance = 0;
            for (double[] mfccs : allMFCCs) variance += Math.pow(mfccs[coeff] - mean, 2);
            result[NUM_COEFFS + coeff] = Math.sqrt(variance / allMFCCs.length);
        }

        // energy mean/std (2 values)
        double energyMean = average(frameEnergies);
        result[26] = energyMean;
        result[27] = stdDev(frameEnergies, energyMean);

        // pitch mean/std (2 values)
        double pitchMean = average(framePitches);
        result[28] = pitchMean;
        result[29] = stdDev(framePitches, pitchMean);

        // zero crossing rate (1 value)
        result[30] = zcr;

        return result;
    }

    public static void main(String[] args) throws Exception {
        String testFile = "/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV/1001_DFA_ANG_XX.wav";
        double[] vector = extract(testFile);
        System.out.println("Final feature vector (" + vector.length + " values):");
        for (int i = 0; i < vector.length; i++) {
            System.out.println(i + ": " + vector[i]);
        }
    }
}