package math;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;  //library for map() function

public class Matrix {

    private final double[][] data;
    private final int rows;
    private final int cols;

    // creates matrix filled with zeros
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    // creates a matrix from a 2D array (deep copy)
    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data.length > 0 ? data[0].length : 0;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = data[i][j];
            }
        }
    }
    public int getRows() {
        return rows;

    }
    public int getCols() {
        return cols;
    }

    // reads one value at position (row, col)
    public double get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IllegalArgumentException("index out of bounds: (" + row +", "+ col +")");
        }
        return data[row][col];
    }

    // writes one value at position (row, col)
    public void set(int row, int col, double value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IllegalArgumentException("index out of bounds: (" + row + ", " + col + ")");
        }
        data[row][col] = value;
    }

    // matrix multiplication
    public Matrix mul(Matrix other) {
        if (this.cols != other.rows) {      //can't multiply two matrices with this property
            throw new IllegalArgumentException("shape mismatch");
        }
        Matrix result = new Matrix(this.rows, other.cols);

        //sum each dot product
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // individual addition - needed for back propagation
    public Matrix elementAdd(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("shape mismatch, we need identical matrices)");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // individual subtraction also needed for back propagayyon
    public Matrix elementSubtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("shape mismatch, we need identical matrices)");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    // individual multiplication for backpropagation
    public Matrix elementMultiply(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("shape mismatch, we need identical matrices)");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    // applies a function to every element
    public Matrix map(DoubleUnaryOperator f) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = f.applyAsDouble(this.data[i][j]);
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    // scalar multiplication
    public Matrix scale(double factor) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * factor;
            }
        }
        return result;
    }

    // fills matrix with random values between min and max
    public void randomize(double min, double max) {
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = min + (max - min) * random.nextDouble();
            }
        }
    }

    // returns a deep copy
    public Matrix copy() {
        Matrix copied = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                copied.data[i][j] = this.data[i][j];
            }
        }
        return copied;
    }

    // prints the matrix aesthetically
    @Override
    public String toString() {
        StringBuilder something = new StringBuilder();       //better than just String because we don't throw away each previous string
        for (int i = 0; i < rows; i++) {
            something.append("[");
            for (int j = 0; j < cols; j++) {
                something.append(String.format("%8.4f", data[i][j]));   //format is for equal decimals and digits on all rows
                if (j < cols - 1) something.append(", ");
            }
            something.append(" ]\n");
        }
        return something.toString();
    }
}