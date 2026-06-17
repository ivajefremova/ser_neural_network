# Claude Notes — Speech Emotion Recognition Project

This file is for Claude Code to read at the start of a new session to get full context.
It is gitignored and never pushed to GitHub.

---

## What This Project Is

A speech emotion recognition system built entirely from scratch in Java — no ML libraries.
It takes a `.wav` audio file, extracts 40 MFCC features from it, feeds them through a custom
neural network, and predicts which of 6 emotions the speaker is expressing.

University project for Programming 2.
Team: Iva Jefremova (branch: `iva`) and Oskar Podkowa (branch: `Oskar`).

Dataset: CREMA-D — 7,442 `.wav` files from 91 actors, 6 emotions.
Filename format: `1001_DFA_ANG_XX.wav` — emotion is the 3rd segment split by `_`.

---

## Emotion Labels

Defined in `config/Emotion.java` on the `iva` branch:
```
HAPPY   = index 0
SAD     = index 1
ANGRY   = index 2
DISGUST = index 3
FEAR    = index 4
NEUTRAL = index 5
```

**Important conflict:** Oskar's `Emotion.java` uses `FEARFUL` instead of `FEAR`, and has no
helper methods. Iva's version has `findIndex()`, `findEmotion(int)`, and `getLabel()`. When
merging, keep Iva's version and update Oskar's `LabelParser.java` to map `"FEA"` → `FEAR`
instead of `FEARFUL`.

---

## Actual Project Structure (current state)

```
src/main/java/
├── config/
│   ├── Config.java              done — global constants
│   └── Emotion.java             done — 6 emotion enum with helper methods
├── math/
│   └── Matrix.java              done — all matrix math
├── network/
│   └── NeuralNetwork.java       done — forward, backward, predict
├── feature/                     empty — MFCCExtractor.java goes here
├── training/                    empty — Trainer.java goes here
└── evaluation/                  empty — Evaluator.java goes here
```

Oskar's branch (`Oskar`) has these additional files not yet merged:
```
src/main/java/
├── audio/
│   ├── AudioLoader.java         done — scans folder, loads .wav files into LabeledAudio list
│   ├── LabelParser.java         done — parses emotion from filename
│   ├── LabeledAudio.java        done — container: filePath + Emotion label
│   └── DatasetSplitter.java     done — splits by actor ID, 80% train / 20% test
└── features/                    NOTE: Oskar uses "features" (plural), Iva uses "feature" (singular)
    └── MFCCExtractor.java       INCOMPLETE — reads WAV bytes but stops before MFCC math
```

When merging: decide on `feature` vs `features` folder name and rename consistently.

---

## Config.java Constants

```java
INPUT_SIZE   = 40      // number of MFCC features per audio clip
HIDDEN_SIZE  = 64      // neurons in hidden layer
OUTPUT_SIZE  = 6       // one per emotion
LEARNING_RATE = 0.01
EPOCHS       = 500
DATA_PATH    = "src/main/resources/data/"
MODEL_PATH   = "src/main/resources/model.txt"
```

---

## Matrix.java — What It Has

mul(Matrix)             — matrix multiplication
elementAdd(Matrix)      — element-wise addition
elementSubtract(Matrix) — element-wise subtraction
elementMultiply(Matrix) — element-wise multiplication
map(DoubleUnaryOperator)— apply any function to every element
transpose()             — flip rows and cols
scale(double)           — multiply every element by a scalar
randomize(min, max)     — fill with random values
copy()                  — deep copy
get(row, col)           — read one value
set(row, col, value)    — write one value
toString()              — pretty print

---

## NeuralNetwork.java — Complete

### Fields
```
weightsHiddenInput  [40 × 64]  — Layer 1 weights
weightsHiddenOutput [64 × 6]   — Layer 2 weights
biasHidden          [1  × 64]  — Layer 1 bias
biasOutput          [1  × 6]   — Layer 2 bias
lastInput           [1  × 40]  — saved during forward() for use in backward()
lastHidden          [1  × 64]  — saved during forward() for use in backward()
```

### forward(Matrix inputMatrix) → Matrix [1×6]
Runs input through the network, returns 6 probabilities (softmax output).
Also saves `lastInput` and `lastHidden` for backward pass.

Flow:
```
[1×40] × [40×64] + [1×64] → tanh → [1×64]  (hidden layer)
[1×64] × [64×6]  + [1×6]  → softmax → [1×6] (output layer)
```

### backward(Matrix resultMatrix, int label)
`resultMatrix` = output of forward() = the 6 probabilities the network predicted.
`label` = the correct emotion index (0-5), known from the filename during training.

Variable naming convention Iva chose: twin+2 for gradient matrices.
- `whichEmotion`       — one-hot target vector [1×6], 1.0 at position `label`
- `errorMatrix`        — resultMatrix - whichEmotion = output error [1×6]
- `weightsHiddenOutput2` — gradient for weightsHiddenOutput [64×6]
- `biasOutput2`        — gradient for biasOutput [1×6]
- `layer1Error`        — error pushed back through Layer 2 weights [1×64]
- `tanhDerivative`     — slope of tanh at each hidden neuron: 1 - h² [1×64]
- `beforeTanh`         — layer1Error after tanh correction [1×64]
- `weightsHiddenInput2`— gradient for weightsHiddenInput [40×64]
- `biasHidden2`        — gradient for biasHidden [1×64]

### predict(Matrix inputMatrix) → int
Runs forward(), returns the index of the highest probability.
This is what the network thinks the emotion is.
NOT used during training — only used after training on new audio.

### softmax(Matrix) — private helper
Converts raw output numbers into probabilities that sum to 1.
Uses e^x so all values are positive, then divides each by the total.

---

## The Full Pipeline (planned)

```
Main.java
  │
  ├── AudioLoader.loadDataset(Config.DATA_PATH)
  │       → List<LabeledAudio>  (all 7442 files with labels)
  │
  ├── DatasetSplitter.split(dataset)
  │       → trainList (80% of actors)
  │       → testList  (20% of actors — same actor never in both)
  │
  ├── Trainer.train(network, trainList)
  │       → for each file: MFCCExtractor → [1×40] Matrix
  │       → forward() + backward() for each sample
  │       → repeat for EPOCHS iterations
  │       → prints loss every 50 epochs
  │
  ├── Evaluator.evaluate(network, testList)
  │       → for each file: MFCCExtractor → predict()
  │       → compare to true label
  │       → print accuracy per emotion + overall
  │
  └── Play one audio file through speakers and show predicted emotion
        → uses javax.sound.sampled (already in Java, no extra library)
        → AudioSystem.getAudioInputStream + Clip
```

---

## Files Still To Write

### feature/MFCCExtractor.java
Takes a `.wav` file path, returns a `Matrix [1×40]`.
Oskar started this — he reads the WAV bytes and converts to `double[]` samples scaled to [-1, 1].
What's missing: the actual MFCC computation (FFT, mel filterbank, DCT).
This is the hardest remaining piece. Needs `commons-math3` which is in Oskar's pom.xml.

### training/Trainer.java
Loops through trainList, calls MFCCExtractor on each file, calls forward() then backward().
Does this for EPOCHS iterations. Prints loss every 50 epochs.
The training loop does NOT go in NeuralNetwork.java — NeuralNetwork is pure math only.

### evaluation/Evaluator.java
Loops through testList, calls MFCCExtractor then predict() on each file.
Compares predicted label to true label. Prints accuracy per emotion and overall.
Breaking it down per emotion is useful — shows which emotions the network is best/worst at.

### Main.java
Entry point. Calls everything in order. Also plays a sample .wav file through the speakers
at the end using javax.sound.sampled so the professor can hear it during the presentation.

---

## Decisions Made During This Session

- `train()` does NOT go in NeuralNetwork.java — it goes in Trainer.java. NeuralNetwork stays pure math.
- The `test/` source folder is not needed — DatasetSplitter handles the split in memory at runtime from one folder.
- Gradient variables are named as twin+2 (e.g. weightsHiddenOutput → weightsHiddenOutput2).
- `backward()` parameter was renamed from `predicted` to `resultMatrix` by Iva.
- `predict()` is separate from `backward()` — predict is for after training on new clips, backward is for during training with known labels.
- No model save/load yet (Config.MODEL_PATH exists but nothing implements it — optional future feature).
- Presentation plan: show console output of training loss decreasing, final accuracy, then play one .wav file and print the predicted emotion.

---

## Merging Order

1. Merge Oskar's branch into iva
2. Reconcile Emotion enum (FEARFUL → FEAR in LabelParser)
3. Reconcile folder name (features vs feature)
4. Finish MFCCExtractor
5. Write Trainer.java
6. Write Evaluator.java
7. Write Main.java

---

## pom.xml Notes

Iva's pom.xml uses Java 26, no external dependencies.
Oskar's pom.xml uses Java 25 and adds `commons-math3` (needed for MFCC math).
When merging: keep Java 26, keep commons-math3 dependency.

---

## Presentation Notes

Professor wants to see:
- Training process live (loss printed to console as it trains)
- Final accuracy on test set broken down by emotion
- A .wav file played through speakers with the predicted emotion printed
