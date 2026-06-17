# Speech Emotion Recognition — Built from Scratch in Java

A neural network that listens to a voice recording and predicts which emotion the speaker is expressing.
Built entirely from scratch without machine learning libraries, as a university project for Programming 2.

---

## What It Does

The program takes an audio file as input, extracts numerical features from the sound, passes them through a neural network, and outputs one of six predicted emotions:

- Happy
- Sad
- Angry
- Disgusted
- Fearful
- Neutral

---

## Dataset

This project uses the **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset), available on Kaggle.
It contains 7,442 audio clips from 91 actors expressing the six target emotions.
Only the `.wav` audio files are used — no video.

Filenames follow the format `1001_DFA_ANG_XX.wav`, where the third segment encodes the emotion:
`ANG` = Angry, `HAP` = Happy, `SAD` = Sad, `DIS` = Disgust, `FEA` = Fear, `NEU` = Neutral.

---

## How It Works

### 1. Audio Loading
`AudioLoader.java` scans the dataset folder and loads every `.wav` file into a `LabeledAudio` object — a container that holds the file path and the emotion label parsed from the filename by `LabelParser.java`.

### 2. Dataset Splitting
`DatasetSplitter.java` splits the full dataset into 80% training and 20% testing, grouped by actor ID. This ensures the same speaker's voice never appears in both sets, which gives a fairer accuracy result.

### 3. Feature Extraction
Raw audio samples are not useful for a neural network directly — there are too many of them and they carry too much noise. Instead, `MFCCExtractor.java` computes **MFCC features** (Mel-Frequency Cepstral Coefficients), which compress each audio clip into exactly 40 numbers that capture the tonal and rhythmic qualities of speech. This 40-number vector is the actual input to the neural network.

### 4. The Neural Network
The 40 MFCC features are passed through a feedforward neural network:

```
Input layer     →   40 neurons   (one per MFCC feature)
Hidden layer    →   64 neurons   (learns patterns in the features)
Output layer    →    6 neurons   (one probability per emotion)
```

Every connection between neurons has a **weight** — a number that determines how much influence one neuron has on the next. Between layers, activations use **tanh** (squishes values to -1 to 1). The output layer uses **softmax** to convert raw scores into 6 probabilities that sum to 1.

### 5. Training
The network is shown thousands of labeled audio clips. For each one:
- It runs the clip through the network (**forward pass**) and gets a prediction
- It compares the prediction to the correct answer and measures how wrong it was
- It runs **backpropagation** — working backwards through the network to figure out how much each weight contributed to the mistake, then nudging every weight slightly in the direction that reduces the error

This repeats 500 times over the full training set. The **learning rate** (0.01) controls how large each nudge is — small enough that the network learns gradually rather than overcorrecting.

### 6. Evaluation
`Evaluator.java` runs the trained network on the held-out test set and reports accuracy per emotion and overall, so you can see which emotions the network recognises best.

### 7. Prediction
Once trained, the network takes a new audio clip it has never heard, extracts its features, and predicts the emotion in milliseconds.

---

## Project Structure

```
speech_emotion_recognition/
├── pom.xml
├── README.md
└── src/
    └── main/
        ├── java/
        │   ├── audio/
        │   │   ├── AudioLoader.java         scans folder, returns list of LabeledAudio
        │   │   ├── LabelParser.java         parses emotion from filename (e.g. ANG → ANGRY)
        │   │   ├── LabeledAudio.java        container: file path + emotion label
        │   │   └── DatasetSplitter.java     80/20 split by actor ID
        │   ├── config/
        │   │   ├── Config.java              global constants (layer sizes, learning rate, paths)
        │   │   └── Emotion.java             enum of the six emotions with index helpers
        │   ├── evaluation/
        │   │   └── Evaluator.java           accuracy on test set, broken down per emotion
        │   ├── features/
        │   │   ├── MFCCExtractor.java       audio file → [1×40] Matrix of features
        │   │   ├── DatasetBuilder.java      builds feature vectors from audio files
        │   │   ├── FeatureVector.java       container: feature array + emotion label
        │   │   └── FeatureNormalizer.java   normalizes features (fit on train, apply to all)
        │   ├── math/
        │   │   └── Matrix.java              all matrix math — multiply, add, transpose, etc.
        │   ├── network/
        │   │   ├── NeuralNetwork.java       forward pass, backpropagation, predict
        │   │   └── ModelSaver.java          saves/loads trained weights to disk
        │   └── org/example/
        │       └── Main.java                entry point — ties everything together
        └── resources/
            └── data/                        place CREMA-D .wav files here
```

---

## Architecture Details

### Matrix.java
The foundation of the entire project. Since no machine learning libraries are used, all neural network math is done through a custom `Matrix` class. It supports:
- Matrix multiplication (`mul`)
- Element-wise add, subtract, multiply
- Transposition (`transpose`)
- Scalar multiplication (`scale`)
- Applying any function to every element (`map`)
- Random initialisation (`randomize`)

Every layer, weight set, and gradient in the network is represented as a `Matrix`.

### NeuralNetwork.java
Holds the four permanent weight and bias matrices and two matrices saved during the forward pass for use in backpropagation (`lastInput` and `lastHidden`).

**`forward(Matrix input)`** — runs the input through the network and returns a `[1×6]` probability vector. Also saves intermediate values needed by backward.

**`backward(Matrix resultMatrix, int label)`** — takes the output of forward and the correct emotion index. Computes gradients for all four weight/bias matrices and updates them.

**`predict(Matrix input)`** — runs forward and returns the index of the highest probability. Used after training to classify new audio clips.

### MFCCExtractor.java
Takes a `.wav` file path, reads the audio bytes, converts them to samples scaled to `[-1, 1]`, and computes 40 MFCC coefficients using FFT and a mel filterbank. Returns a `Matrix [1×40]` ready to feed into the network.

### Evaluator.java
Loops through the test set, extracts features, calls `predict()`, and compares to the true label. Prints accuracy per emotion and overall accuracy.

---

## Technologies

| Tool | Purpose |
|---|---|
| Java | primary language |
| Maven | dependency management and build |
| javax.sound.sampled | reading .wav files (built into Java) |
| IntelliJ IDEA | IDE |
| GitHub | version control and collaboration |
| CREMA-D (Kaggle) | training and test data |

No machine learning libraries (e.g. Deeplearning4j) are used. All neural network math is implemented from scratch.

---

## Setup and Run

### Prerequisites
- Java 26 or higher
- Maven

### Clone and build
```bash
git clone https://github.com/ivajefremova/ser_neural_network.git
cd speech_emotion_recognition
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

University project — Programming 2.
