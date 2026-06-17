package training;

import features.FeatureVector;
import math.Matrix;
import config.Config;
import network.NeuralNetwork;




import java.util.Collections;
import java.util.List;

public class Training {
    public static Matrix tomatrix(double[] vector){
        Matrix mat = new Matrix(1, vector.length);
        for (int i = 0; i< vector.length; i++){
            mat.set(0, i, vector[i]);
        }
        return mat;
    }

    public static void train(NeuralNetwork network, List<FeatureVector> training_data){

        for (int epoch= 0; epoch < Config.EPOCHS; epoch++){

            Collections.shuffle(training_data);

            int correct = 0;
            double totalLoss = 0;

            for(FeatureVector fv: training_data){
                Matrix input = tomatrix(fv.getFeatures());
                int label = fv.getLabel().ordinal();

                Matrix output = network.forward(input);

                int predicted = 0;
                for(int i =0; i < Config.OUTPUT_SIZE; i++){
                    if (output.get(0, i)> output.get(0, predicted)){
                        predicted = i;
                    }
                }
                if ( predicted == label){
                    correct++;
                }

                double prob = output.get(0, label);
                totalLoss += -Math.log(prob + 1e-10);

                network.backward(output, label);
            }

            double accuracy = (double) correct / training_data.size();
            double avgLoss = totalLoss / training_data.size();

            System.out.printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%%%n",
                    epoch, avgLoss, accuracy * 100);
        }
    }

}
