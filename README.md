# pyAudioTest

## Prerequisites

- Python 3.x
- FFmpeg: This tool is used for audio preprocessing. Please ensure it is installed and accessible in your system's PATH.
  - On macOS: `brew install ffmpeg`
  - On Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
  - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To classify an audio file, run the `audio_classifier.py` script:

```bash
python src/audio_classifier.py path/to/your/audiofile.wav
```

**Arguments:**

*   `input_audio_file`: (Required) Path to the input audio file (e.g., WAV, MP3, MP4, MOV).
*   `--model_path`: (Optional) Path to a specific pyAudioAnalysis SVM model file. Defaults to the general speech/music model included with pyAudioAnalysis.
*   `--output_dir`: (Optional) Directory where the output CSV and `.dcsx` files will be saved. Defaults to the same directory as the input audio file.

**Example:**

Assuming you are in the project root directory:
```bash
python src/audio_classifier.py sample_audio/test_sample.wav --output_dir output_files
```
*(Ensure you create an `output_files` directory first if you use this example, or point to an existing one. The `sample_audio/test_sample.wav` is provided in the repository.)*

## Outputs

The script generates two files in the specified output directory (or the input file's directory by default):

1.  **`<input_filename>_logic_pro.csv`**:
    *   A CSV file formatted for import as markers into Logic Pro X.
    *   Columns: `Marker,Position,Name`
    *   `Position` is the start timecode of the segment (`HH:MM:SS.mmm`).
    *   `Name` is the classification label (e.g., "speech", "music").

2.  **`<input_filename>.dcsx`**:
    *   A JSON file containing structured information about the classified segments.
    *   Includes the source audio filename and a list of cues, each with:
        *   `CueID`
        *   `StartTimecode`
        *   `Label` (classification)

## Running Tests

The project includes unit tests to verify basic functionality. Ensure you have installed dependencies first.

To run the tests, navigate to the project root directory and run:

```bash
python -m unittest discover -s tests
```