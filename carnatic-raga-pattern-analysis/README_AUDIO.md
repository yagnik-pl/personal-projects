# Carnatic Raga Identifier - Audio Feature Extraction & Prediction

## Overview
This guide explains how to use the audio feature extraction and prediction functionality to identify Carnatic ragas from .wav audio files.

## Setup

### 1. Install Dependencies
Make sure you have librosa installed:
```bash
pip install librosa
```

### 2. File Structure
```
Carnatic-project/
├── carnatic-raga-identifier.py    # Main ML model and audio extraction code
├── Dataset.csv                     # Training dataset
├── models.joblib                   # Trained model bundle (after training)
└── test-data/                      # Folder for test .wav files
    └── *.wav                       # Your audio files to test
```

## Usage

### Mode 1: Train the Model
```bash
python carnatic-raga-identifier.py
```
This trains both LightGBM and XGBoost models and saves them to `models.joblib`.

### Mode 2: Test on Audio Files
```bash
python carnatic-raga-identifier.py test
```
or specify a custom test folder:
```bash
python carnatic-raga-identifier.py test path/to/audio/folder
```

This will:
1. Load all .wav files from the test-data folder
2. Extract MFCC features from each audio file
3. Make predictions using the trained model
4. Display results in a table format

## Audio Feature Extraction Pipeline

### Step 1: Load Audio
```python
y, sr = librosa.load(file_path, sr=22050)
```

### Step 2: Preprocess Audio
- **Trim silence**: Remove leading/trailing silence
- **Normalize**: Scale audio to [-1, 1] range
- **Fix duration**: Pad/trim to 5 seconds for consistency

### Step 3: Extract MFCC Features
```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=19)
```
Output shape: (19, time_frames)

### Step 4: Convert to Feature Vector
```python
features = np.mean(mfcc, axis=1)  # Average across time
```
Output shape: (19,)

### Step 5: Match Dataset Format
Convert to dictionary with keys: `mfcc0, mfcc1, ..., mfcc18`

## Example Usage

### Using the provided functions:

```python
import joblib
from carnatic_raga_identifier import predict_from_audio

# Load model
bundle = joblib.load("models.joblib")

# Predict from audio file
result = predict_from_audio("test-data/sample.wav", bundle)

print(result)
# Output:
# {
#     'file': 'test-data/sample.wav',
#     'prediction': 'Bhairavi',
#     'confidence': 0.85,
#     'margin': 0.65,
#     'status': 'success'
# }
```

### Direct feature extraction:

```python
from carnatic_raga_identifier import extract_features_for_model

features = extract_features_for_model("test-data/sample.wav")
print(features)  # {'mfcc0': 0.123, 'mfcc1': -0.456, ...}
```

## Output Format

When testing audio files, you get a table with:
- **File**: Audio filename
- **Prediction**: Predicted Carnatic raga class
- **Confidence**: Confidence score (0-1)
- **Margin**: Margin between top 2 classes
- **Status**: 'success' or error message

Example output:
```
File                                          Prediction           Confidence  Status
-------------------------------------------------------------------------------------------------------
sample1.wav                                   Bhairavi             0.8806      success
sample2.wav                                   Kalyani              0.9124      success
invalid_file.wav                              error                0.0000      error: Invalid audio format
```

## Supported Audio Formats
- WAV (.wav)
- Other formats supported by librosa (mp3, flac, ogg, etc.)

## Model Parameters

- **Sample Rate**: 22050 Hz
- **Duration**: 5 seconds (fixed)
- **MFCC Coefficients**: 19
- **Classification Classes**: 8 Carnatic ragas (when using filtered dataset)
- **Confidence Threshold**: 0.575 (for rejection)
- **Margin Threshold**: 0.25 (margin between top 2 predictions)

## Troubleshooting

### Error: "No .wav files found"
- Make sure your audio files are in the `test-data` folder
- Files must have `.wav` extension
- Use `test path/to/folder` to specify custom location

### Error: "Invalid audio format"
- Ensure the audio file is valid and readable
- Try with a different audio file
- Install ffmpeg for better audio format support: `pip install librosa[audioread]`

### Low prediction accuracy
- Ensure audio quality is good
- Audio should be 5+ seconds long
- Try different audio files for testing
- Check that the model was properly trained

## Next Steps

1. Place your test audio files in `test-data/` folder
2. Run `python carnatic-raga-identifier.py test`
3. Check the results table for predictions
4. Adjust thresholds if needed for your use case
