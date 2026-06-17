package network;

import math.Matrix;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ModelSaver {

    public static void save(NeuralNetwork nn, String path) throws IOException {
        Files.createDirectories(Paths.get(path).getParent());
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));

        writeMatrix(writer, "weightsHiddenInput",  nn.getWeightsHiddenInput());
        writeMatrix(writer, "weightsHiddenOutput", nn.getWeightsHiddenOutput());
        writeMatrix(writer, "biasHidden", nn.getBiasHidden());
        writeMatrix(writer, "biasOutput", nn.getBiasOutput());

        writer.close();
    }


    public static NeuralNetwork load(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        NeuralNetwork nn = new NeuralNetwork();

        nn.setWeightsHiddenInput(readMatrix(reader));
        nn.setWeightsHiddenOutput(readMatrix(reader));
        nn.setBiasHidden(readMatrix(reader));
        nn.setBiasOutput(readMatrix(reader));

        reader.close();
        return nn;
    }


    private static void writeMatrix(BufferedWriter writer, String label, Matrix m) throws IOException {
        writer.write(label + " " + m.getRows() + " " + m.getCols());
        writer.newLine();
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                writer.write(Double.toString(m.get(i, j)));
                if (j < m.getCols() - 1) writer.write(" ");
            }
            writer.newLine();
        }
    }

    private static Matrix readMatrix(BufferedReader reader) throws IOException {
        String[] header = reader.readLine().split(" ");
        int rows = Integer.parseInt(header[1]);
        int cols = Integer.parseInt(header[2]);

        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            String[] values = reader.readLine().split(" ");
            for (int j = 0; j < cols; j++) {
                m.set(i, j, Double.parseDouble(values[j]));
            }
        }
        return m;
    }
}
