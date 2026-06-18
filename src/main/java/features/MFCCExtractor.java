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

//WAV files - essentially a long list of numbers that represent air pressure over time

// At 16,000 Hz -sample rate it records 16,000 numbers per second. A 3-second clip = 48,000 numbers.
// Each number is stored as a 16-bit integer, meaning it can be anywhere from -32,768 to +32,767.
// Negative = air pressure below baseline, positive = above. Zero = silence.
// The raw file bytes look like:
//  ... 0x00 0x1A  0x00 0x2F  0xFF 0xE1 ...
//  Each pair of bytes is one sample — one snapshot of the waveform.

public class MFCCExtractor {

    private static final int FRAME_SIZE = 512;
    private static final int HOP_SIZE = 256;
    private static final int NUM_FILTERS = 33;
    private static final int NUM_COEFFS = 13;

    private static final int MIN_PITCH_HZ = 80;
    private static final int MAX_PITCH_HZ = 400;

    private static double hzToMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700.0);
    }   //from herz to Melofdy math

    private static double melToHz(double mel) {
        return 700 * (Math.pow(10, mel / 2595.0) - 1);
    }   //mel to herz

    //makes an array of representations of waveform from the bytestream of the audio -1 to 1
    public static double[] readWAV(String filePath) throws Exception{
        File file = new File(filePath);
        AudioInputStream ais = AudioSystem.getAudioInputStream(file);   //part of AudioSystem class in java , gets a stream if bytes

        byte[] buffer = ais.readAllBytes();     //reads the remaining bytes
        double[] samples = new double[buffer.length / 2];   //list of doubles of remaining bytes /2

        //every 2 bytes = one audio sample - 1 integer
        //shifting to get the number where it should be
        for (int i = 0; i < buffer.length; i += 2) {
            int sample = (buffer[i+1] << 8) | (buffer[i] & 0xFF);
            samples[i/2] = sample / 32768.0;
        }

        ais.close();
        return samples;    //array of elements that represent a waveform at a time from -1 to 1
    }

    //chops that long samples array into small overlapping chunks frames
    private static double[][] frameSignal(double[] samples){
        int numFrames = (samples.length - FRAME_SIZE) / HOP_SIZE + 1;
        double[][] frames = new double[numFrames][FRAME_SIZE];
        for (int i = 0; i < numFrames; i++){
            int start = i * HOP_SIZE;
            System.arraycopy(samples, start, frames[i], 0, FRAME_SIZE);
        }
        return frames;      //2D array frames[numFrames][frame size]
    }

    // it multiplies each frame by a bell-shaped curve that smoothly fades to near-zero at both edges
    private static void applyHammingWindow(double[] frame){
        int N = frame.length;
        for (int n = 0; n < N; n++){
            double window = 0.54 - 0.46 * Math.cos(2 * Math.PI * n / (N - 1)); // fixed
            frame[n] = frame[n] * window;
        }
    }

    //
    private static double[] computePowerSpectrum(double[] frame){
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);  //library class from Apache Commons Math
        Complex[] spectrum = fft.transform(frame, TransformType.FORWARD);      //spectrum is complex transformed array with complex numbers
        //hand it audio samples, it hands back complex numbers representing frequencies.
        //the math behind it ( Fourier Transform algorithm) is handled entirely inside the library.
        double[] res = new double[FRAME_SIZE/2];   //Only the first half is kept. The FFT of a real signal is always symmetric,  bins 256–511 are a mirror of bins 0–255,
        for (int i = 0; i < FRAME_SIZE/2; i++){
            double real = spectrum[i].getReal();
            double imag = spectrum[i].getImaginary();
            res[i] = (real * real) + (imag * imag);
        }
        return res;    //arraty of (real * real) + (imag * imag) of every complex number in half spectrum array
    }

    // The power spectrum has 256 frequency bins spaced evenly — every 31 Hz.
    // human hearing doesn't work that way - very sensitive to differences in low frequencies
    // but bad at distinguishing high frequencies-500 Hz and 600 Hz sound more similar
    // mel scale mimics this — it squishes the high frequencies and spreads out the low ones - to match how humans hear
    private static double[] applyMelFilterbank(double[] powerSpectrum){
        int numFilters = NUM_FILTERS;
        double[] filterbank = new double[numFilters];

        double melMin = hzToMel(0);
        double melMax = hzToMel(Config.SAMPLE_RATE / 2.0);

        double[] melPoints = new double[numFilters + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + i * (melMax - melMin) / (numFilters + 1);
        }
        // convert those points back to Hz, then to FFT bin indices
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
        return filterbank; //array of 33 numbers — each is the total energy in one mel-frequency band.
    }

    //human hearing doesn't perceive loudness linearly either-logarithmically
    //  loud:  [5000, 12000, 800]  →  [8.5,  9.4,  6.7]
    //  quiet: [0.5,  1.2,  0.08]  →  [−0.7, 0.2, −2.5]
    //  The values are now in a much tighter, more manageable range. The network can handle these numbers much more easily than values spanning from 0.001 to 50,000.
    private static double[] applyLog(double[] filterbank) {
        double[] result = new double[filterbank.length];
        for (int i = 0; i < filterbank.length; i++) {
            result[i] = Math.log(filterbank[i] + 1e-10);
        }
        return result;    //array of log + 1e-10 of the filterbank, 33 numbers representing energy in 33 mel bands
    }

    //
    private static double[] applyDCT(double[] logFilterbank) {
        FastCosineTransformer dct = new FastCosineTransformer(DctNormalization.STANDARD_DCT_I);
        double[] result = dct.transform(logFilterbank, TransformType.FORWARD);
        return Arrays.copyOfRange(result, 0, NUM_COEFFS);
    }

    // energy of one frame
    private static double computeFrameEnergy(double[] frame) {
        double sum = 0;
        for (double v : frame) {
            sum += v * v;    //squaring does two things: makes all values positive (a negative sample is still energy), and emphasizes louder samples more
        }
        return sum / frame.length;  //average energy per sample so frames of different lengths would be comparable
    }

    // pitch estimate of one frame via autocorrelation
    private static double estimatePitch(double[] frame) {
        int minLag = Config.SAMPLE_RATE / MAX_PITCH_HZ;
        int maxLag = Config.SAMPLE_RATE / MIN_PITCH_HZ;

        double bestCorrelation = -1;
        int bestLag = minLag;

        //if a signal repeats every N samples, then shifting it by N and multiplying it with
        // the original gives a big number — they line up. Shift by the wrong
        //  amount and they cancel out - autocorrelation.
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
        //If the best lag is 80 samples:
        // 16000 / 80 = 200 Hz  ←  estimated pitch of this frame
        return (double) Config.SAMPLE_RATE / bestLag;
    }

    // zero crossing rate over the whole file - if the sound is smooth or no, zero crossing means from negative to pos values in our array
    //neutral and sad speech tends to be smoother, fearful and disgusted speech often breathier — so ZCR adds a signal the MFCCs alone don't fully capture.
    private static double computeZCR(double[] samples) {
        int crossings = 0;
        for (int i = 1; i < samples.length; i++) {
            boolean signChanged = (samples[i] >= 0 && samples[i-1] < 0) || (samples[i] < 0 && samples[i-1] >= 0);
            if (signChanged) crossings++;
        }
        return (double) crossings / samples.length;        //average amount of crossings
    }

    //computes average noth spec
    private static double average(double[] values) {
        double sum = 0;
        for (double v : values) sum += v;
        return sum / values.length;
    }

    //small std dev = pitch barely moved (monotone, flat — think sad/neutral).
    //large std dev = pitch jumped around a lot (expressive — think angry/happy).
    private static double stdDev(double[] values, double mean) {
        double variance = 0;
        for (double v : values) variance += Math.pow(v - mean, 2);   // for each value, find how far it is from the mean and square it
        return Math.sqrt(variance / values.length);
    }

    //calls all the other methods in order and assembles the final 31-value feature vector
    public static double[] extract(String filePath) throws Exception {
        double[] rwav = readWAV(filePath);
        double[][] frames = frameSignal(rwav);

        double[][] allMFCCs = new double[frames.length][NUM_COEFFS];
        double[] frameEnergies = new double[frames.length];
        double[] framePitches = new double[frames.length];

        //process each frame
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

        return result; //final array passed to neural  network and feature vector
    }
    // [0–12]  MFCC means
    //  [13–25] MFCC std devs
    //  [26]    energy mean
    //  [27]    energy std dev
    //  [28]    pitch mean
    //  [29]    pitch std dev
    //  [30]    ZCR


    //test for a file
    public static void main(String[] args) throws Exception {
        String testFile = "/Users/oskar/Documents/NeuroNetProject/Data/AudioWAV/1001_DFA_ANG_XX.wav";
        double[] vector = extract(testFile);
        System.out.println("Final feature vector (" + vector.length + " values):");
        for (int i = 0; i < vector.length; i++) {
            System.out.println(i + ": " + vector[i]);
        }
    }
}