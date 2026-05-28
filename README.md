### Speech Emotion Recognition — Built from Scratch in Java

A neural network that listens to a voice recording and predicts which emotion the speaker is expressing.
Built entirely from scratch without machine learning libraries, as a university project for Programming 2.

---

## What It Does

The program takes an audio file as input, extracts numerical features from the sound, passes them through a neural network, and outputs one of six predicted emotions:

- Happy
- Sad
- Angry
- Fearful
- Disgusted
- Neutral

---

## Dataset

This project uses the **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset), available on Kaggle.
It contains 7,442 audio clips from 91 actors expressing the six target emotions.
Only the audio files are used — no video.

---

## How It Works

### 1. Audio Parsing
The raw `.wav` file is read from disk and decoded into an array of audio samples — essentially a long list of numbers representing sound pressure over time.

### 2. Feature Extraction
Raw audio samples are not useful for a neural network directly — there are too many and they carry too much noise. Instead, we extract **MFCC features** (Mel-Frequency Cepstral Coefficients), which compress the audio into a short vector of numbers (around 40) that capture the tonal and rhythmic qualities of speech. This is the standard feature set used in speech recognition.

### 3. The Neural Network
The feature vector is passed through a feedforward neural network with three layers:

```
Input layer      →   40 neurons   (one per MFCC feature)
Hidden layer     →   64 neurons   (learns patterns)
Output layer     →    6 neurons   (one per emotion)
```

Each connection between neurons has a **weight** — a number that determines how much influence one neuron has on the next. These weights start as random values and are adjusted during training.

### 4. Training
The network is shown thousands of labeled audio clips. For each one it makes a prediction, compares it to the correct answer, measures how wrong it was, and adjusts all its weights slightly in the direction that would have made it more correct. This process — called **backpropagation** — repeats tens of thousands of times until the network becomes accurate.

### 5. Prediction
Once trained, the network can take a new audio clip it has never heard, extract its features, and predict the emotion in milliseconds.

---

## Project Structure

```
SER_NeuralNetwork/
├── pom.xml                          maven build config and dependencies
├── README.md
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   ├── audio/
│   │   │   │   ├── AudioParser.java         reads .wav files into sample arrays
│   │   │   │   └── FeatureExtractor.java    converts samples to MFCC feature vectors
│   │   │   ├── config/
│   │   │   │   ├── Config.java              global constants (layer sizes, learning rate, paths)
│   │   │   │   └── Emotion.java             enum of the six emotion labels
│   │   │   ├── evaluation/
│   │   │   │   └── Evaluator.java           measures accuracy on test data
│   │   │   ├── features/
│   │   │   │   └── MFCCExtractor.java       MFCC computation logic
│   │   │   ├── math/
│   │   │   │   └── Matrix.java              all matrix math (multiply, add, transpose, etc.)
│   │   │   ├── network/
│   │   │   │   └── NeuralNetwork.java       forward pass — predicts from input
│   │   │   ├── training/
│   │   │   │   └── Trainer.java             backpropagation and weight updates
│   │   │   └── Main.java                    entry point — ties everything together
│   │   └── resources/
│   └── test/
│       └── java/                            unit tests
└── test/
```

---

## Architecture Details

### Matrix.java
The foundation of the entire project. Since no machine learning libraries are used, all neural network math is done through a custom `Matrix` class that supports dot product multiplication, element-wise operations, transposition, scalar scaling, and function mapping. Every layer, weight set, and gradient in the network is represented as a `Matrix`.

### NeuralNetwork.java
Holds the four permanent weight and bias matrices. Runs the **forward pass**: takes an input matrix, multiplies through each layer, adds biases, applies the activation function (`tanh`), and returns a 1×6 output matrix of emotion scores.

### Trainer.java
Runs the **backward pass** (backpropagation). Compares the network output to the correct label, calculates error, propagates it back through the layers, and nudges each weight in the direction that reduces the error. Uses a configurable **learning rate** to control how large each nudge is.

### AudioParser.java
Reads `.wav` files from disk using Java's standard `AudioInputStream`. Returns the raw PCM samples as a `double[]`.

### FeatureExtractor.java
Applies the MFCC algorithm to the raw samples, returning a `double[]` of ~40 values ready to be wrapped in a Matrix and fed to the network.

### Evaluator.java
Runs the trained network on the test split of the dataset and reports accuracy per emotion and overall.

---

## Technologies

| Tool | Purpose |
|---|---|
| Java 17 | primary language |
| Maven | dependency management and build |
| IntelliJ IDEA | IDE |
| GitHub | version control and collaboration |
| CREMA-D (Kaggle) | training and test data |

No machine learning libraries (e.g. Deeplearning4j) are used. All math is implemented from scratch.

---

## Build and Run

### Prerequisites
- Java 17 or higher
- Maven
- IntelliJ IDEA (recommended)

### Setup
```bash
git clone https://github.com/your-repo/ser_neural_network.git
cd SER_NeuralNetwork
mvn install
```

### Place the dataset
Download CREMA-D from Kaggle and place the `.wav` files under:
```
src/main/resources/data/
```

### Run
```bash
mvn exec:java -Dexec.mainClass="Main"
```

---

## Team
Iva Jefremova
Oskar Podkowa

---

University project, built as an introduction to neural networks and object-oriented design.
