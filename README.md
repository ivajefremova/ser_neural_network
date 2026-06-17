# Speech Emotion Recognition — Built from Scratch in Java

A neural network that listens to a voice recording and predicts which emotion the speaker is expressing.
Built entirely from scratch without machine learning libraries, as a university project for Programming 2.

---

## What It Does

The program takes an audio file as input, extracts numerical features from the sound, passes them through a neural network, and outputs one of six predicted emotions:

- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad

---

## Dataset

This project uses the **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset), available on Kaggle.
It contains 7,442 audio clips from 91 actors expressing the six target emotions.
Only the `.wav` audio files are used — no video.

Filenames follow the format `1001_DFA_ANG_XX.wav`, where the third segment encodes the emotion:
`ANG` = Angry, `HAP` = Happy, `SAD` = Sad, `DIS` = Disgust, `FEA` = Fearful, `NEU` = Neutral.

---

## How It Works

### 1. Audio Loading
`AudioLoader.java` scans the dataset folder and loads every `.wav` file into a `LabeledAudio` object — a container that holds the file path and the emotion label parsed from the filename by `LabelParser.java`.

### 2. Dataset Splitting
`DatasetSplitter.java` splits the full dataset into 80% training and 20% testing, grouped by actor ID. This ensures the same speaker's voice never appears in both sets, which gives a fairer accuracy result.

### 3. Feature Extraction
Raw audio samples are not useful for a neural network directly — there are too many of them and they carry too much noise. Instead, `MFCCExtractor.java` computes a **31-feature vector** per audio clip:

| Features | Count |
|---|---|
| MFCC coefficient means (13 coefficients) | 13 |
| MFCC coefficient standard deviations | 13 |
| Frame energy mean + std | 2 |
| Pitch mean + std (via autocorrelation) | 2 |
| Zero crossing rate | 1 |
| **Total** | **31** |

### 4. Normalisation
`FeatureNormalizer.java` standardises all 31 features to zero mean and unit variance, fit on the training set and applied to both training and test. The normalisation parameters are saved to disk so they can be reused for single-file prediction without retraining.

### 5. The Neural Network
The 31 features are passed through a feedforward neural network:

```
Input layer     →   31 neurons   (one per feature)
Hidden layer    →   64 neurons   (learns patterns in the features)
Output layer    →    6 neurons   (one probability per emotion)
```

Every connection between neurons has a **weight**. Between layers, activations use **tanh** (squishes values to −1 to 1). The output layer uses **softmax** to convert raw scores into 6 probabilities that sum to 1.

### 6. Training
The network is shown thousands of labeled audio clips over 250 epochs. For each clip:
- It runs the clip through the network (**forward pass**) and gets a prediction
- It compares the prediction to the correct answer and measures how wrong it was
- It runs **backpropagation** — working backwards through the network to figure out how much each weight contributed to the mistake, then nudging every weight slightly in the direction that reduces the error

The **learning rate** (0.003) controls how large each nudge is. During training, test accuracy is evaluated every epoch and the best-performing weights are saved and restored at the end, so the final model is the best one seen — not just the last.

### 7. Evaluation
`Evaluator.java` runs the trained network on the held-out test set and reports accuracy per emotion and overall.

### 8. Prediction
Once trained, the model and normalizer are saved to disk. `Predict.java` loads them and classifies any new `.wav` file instantly — no retraining needed. It prints per-emotion probabilities, plays the audio clip, and announces the prediction via text-to-speech.

---

## Project Structure

```
speech_emotion_recognition/
├── pom.xml
├── README.md
├── models/
│   ├── model.txt              saved network weights
│   └── normalizer.txt         saved normalisation parameters
└── src/
    └── main/
        ├── java/
        │   ├── Main.java                  entry point — trains and saves the model
        │   ├── Predict.java               loads trained model, classifies one wav file
        │   ├── audio/
        │   │   ├── AudioLoader.java       scans folder, returns list of LabeledAudio
        │   │   ├── LabelParser.java       parses emotion from filename (e.g. ANG → ANGRY)
        │   │   ├── LabeledAudio.java      container: file path + emotion label
        │   │   └── DatasetSplitter.java   80/20 split by actor ID
        │   ├── config/
        │   │   ├── Config.java            global constants (layer sizes, learning rate, paths)
        │   │   └── Emotion.java           enum of the six emotions
        │   ├── evaluation/
        │   │   └── Evaluator.java         accuracy on test set, broken down per emotion
        │   ├── features/
        │   │   ├── MFCCExtractor.java     audio file → 31-feature vector
        │   │   ├── FeatureNormalizer.java  normalizes features; save/load to disk
        │   │   ├── FeatureVector.java     container: feature array + emotion label
        │   │   └── DatasetBuilder.java    builds feature vectors from audio files
        │   ├── math/
        │   │   └── Matrix.java            all matrix math — multiply, add, transpose, etc.
        │   ├── network/
        │   │   ├── NeuralNetwork.java     forward pass, backpropagation, predict
        │   │   └── ModelSaver.java        saves/loads trained weights to disk
        │   └── training/
        │       └── Training.java          training loop with best-model tracking
        └── resources/
            └── data/
                └── AudioWAV/              place CREMA-D .wav files here
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

**`forward(Matrix input)`** — runs the input through the network and returns a `[1×6]` probability vector.

**`backward(Matrix output, int label)`** — computes gradients for all four weight/bias matrices and updates them via gradient descent.

**`predict(Matrix input)`** — runs forward and returns the index of the highest probability.

### Training.java
Runs the training loop for `Config.EPOCHS` epochs. Each epoch shuffles the training data, runs forward + backward for every sample, and evaluates test accuracy. The best weights seen across all epochs are restored at the end.

### MFCCExtractor.java
Takes a `.wav` file path, reads the audio bytes, and computes 31 features: 13 MFCC means, 13 MFCC standard deviations, frame energy mean/std, pitch mean/std (via autocorrelation), and zero crossing rate.

### FeatureNormalizer.java
Fits mean and standard deviation on the training set, then normalises any feature vector to zero mean and unit variance. Saves and loads parameters from a text file so the same normalisation can be applied at prediction time.

### Evaluator.java
Loops through the test set, calls `predict()`, and compares to the true label. Prints accuracy per emotion and overall accuracy.

---

## Technologies

| Tool | Purpose |
|---|---|
| Java | primary language |
| Maven | dependency management and build |
| commons-math3 | FFT for MFCC computation |
| javax.sound.sampled | reading and playing .wav files (built into Java) |
| macOS `say` | text-to-speech for prediction announcement |
| IntelliJ IDEA | IDE |
| GitHub | version control and collaboration |
| CREMA-D (Kaggle) | training and test data |

No machine learning libraries (e.g. Deeplearning4j) are used. All neural network math is implemented from scratch.

---

## Setup and Run

### Prerequisites
- Java 25 or higher
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
src/main/resources/data/AudioWAV/
```

### Run

**Step 1 — Train the network**
Run `Main.java`. This loads the dataset, trains for 250 epochs, evaluates on the test set, and saves the model and normalizer to `models/`.

**Step 2 — Predict on a single file**
Run `Predict.java` with a `.wav` file path as the program argument:
```
src/main/resources/data/AudioWAV/1001_DFA_ANG_XX.wav
```
It will print per-emotion probabilities, play the audio clip, and announce the predicted emotion out loud.

---

## Team
Iva Jefremova
Oskar Podkowa

University project — Programming 2.
