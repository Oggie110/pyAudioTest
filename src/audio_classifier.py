import os
import subprocess
import shutil  # For copying files if already WAV
import csv
from datetime import datetime, timedelta
import json
import argparse

# Attempt to find the pyAudioAnalysis library path for models
# This is a common way to find package paths, but might need adjustment
try:
    import pyAudioAnalysis
    PYAUDIOANALYSIS_DIR = os.path.dirname(os.path.realpath(pyAudioAnalysis.__file__))
    # Default model path (example, might need to be verified/changed)
    # Common models: svm_rbf_sm (speech-music), svm_rbf_genre (general genres)
    DEFAULT_MODEL_PATH = os.path.join(PYAUDIOANALYSIS_DIR, "data", "svm_rbf_sm")
except ImportError:
    PYAUDIOANALYSIS_DIR = None
    DEFAULT_MODEL_PATH = "pyAudioAnalysis/data/svm_rbf_sm" # Fallback, user might need to set this
    print("Warning: pyAudioAnalysis not found. Model path may be incorrect.")

# pyAudioAnalysis imports
try:
    from pyAudioAnalysis.audioSegmentation import mtFileClassification
    # listOfClasses is used to get the names of the classes from the model
    # This might be specific to how models are trained/stored in pyAudioAnalysis
    # For svm_rbf_sm, the classes are usually ['speech', 'music']
    # We will try to get them dynamically if possible, otherwise use a default.
    model_data_path = os.path.join(PYAUDIOANALYSIS_DIR, "data")
    # This is a common way class names are stored, but might need adjustment
    # if specific model doesn't follow this pattern for listOfClasses.
    # For svm_rbf_sm, it's often implicitly known.
except ImportError:
    print("Error: Failed to import from pyAudioAnalysis. Ensure it is installed.")
    # Provide dummy implementation if import fails, so script can still be loaded.
    def mtFileClassification(*args, **kwargs):
        print("Dummy mtFileClassification: pyAudioAnalysis not imported.")
        return [], [], [] 
    # listOfClasses = ['unknown', 'unknown'] # Fallback

# It's common for the svm_rbf_sm model to have these classes.
# If a different model is used, this list would need to change.
# We will attempt to load this from the model file if possible, but that's more advanced.
# For now, this is a reasonable assumption for the default model.
DEFAULT_CLASS_NAMES = ['speech', 'music']


def convert_to_wav(input_file: str, output_wav_file: str) -> str | None:
    """
    Converts an input audio file to a WAV file (16kHz, mono).
    If the input is already WAV, it copies it to the output path.

    Args:
        input_file: Path to the input audio file.
        output_wav_file: Path to save the output WAV file.

    Returns:
        Path to the WAV file, or None if conversion failed.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    file_ext = os.path.splitext(input_file)[1].lower()

    os.makedirs(os.path.dirname(output_wav_file), exist_ok=True)

    if file_ext == ".wav":
        print(f"Input file '{input_file}' is already a WAV file.")
        try:
            # Check if it's the same file path to avoid copying onto itself
            if os.path.abspath(input_file) != os.path.abspath(output_wav_file):
                shutil.copy(input_file, output_wav_file)
            print(f"Used existing WAV file: '{output_wav_file}'")
            return output_wav_file
        except Exception as e:
            print(f"Error copying WAV file '{input_file}': {e}")
            return None

    print(f"Converting '{input_file}' to WAV format...")
    try:
        # ffmpeg command: -i <input> -ar 16000 (sample rate) -ac 1 (mono) <output.wav>
        # -y overwrites output file if it exists
        result = subprocess.run(
            ["ffmpeg", "-i", input_file, "-ar", "16000", "-ac", "1", "-y", output_wav_file],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully converted '{input_file}' to '{output_wav_file}'")
        # print("FFmpeg output:\n", result.stdout) # Optional: for debugging
        # print("FFmpeg errors:\n", result.stderr) # Optional: for debugging
        return output_wav_file
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion for '{input_file}':")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure it is installed and in your PATH.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during conversion of '{input_file}': {e}")
        return None

# Placeholder for classify_audio_segments
def classify_audio_segments(wav_file_path: str, model_path: str = DEFAULT_MODEL_PATH) -> list:
    """
    Segments the audio file and classifies each segment.
    (Implementation to be added)
    """
    print(f"Attempting to classify segments in: {wav_file_path} using model: {model_path}")
    if not os.path.exists(wav_file_path):
        print(f"Error: WAV file '{wav_file_path}' not found for classification.")
        return []
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".pt"): # .pt for newer PyTorch models
         # Also check for model files like .pkl, .model, etc. if svm_rbf_sm is a generic name
        model_path_abs = os.path.abspath(model_path)
        print(f"Error: Model file '{model_path_abs}' (or related files like .pt, .pkl) not found.")
        print(f"Please ensure the pyAudioAnalysis models are correctly installed or provide a valid model_path.")
        print(f"PYAUDIOANALYSIS_DIR was detected as: {PYAUDIOANALYSIS_DIR}")
        print(f"Tried to find model at: {DEFAULT_MODEL_PATH}")
        # Attempt to list contents of PYAUDIOANALYSIS_DIR/data to help user
        if PYAUDIOANALYSIS_DIR and os.path.exists(os.path.join(PYAUDIOANALYSIS_DIR, "data")):
            print(f"Available models/data in {os.path.join(PYAUDIOANALYSIS_DIR, 'data')}:")
            try:
                for item in os.listdir(os.path.join(PYAUDIOANALYSIS_DIR, "data")):
                    print(f"  - {item}")
            except Exception as e:
                print(f"    Could not list directory: {e}")
        return []
    
    # Actual classification logic using pyAudioAnalysis
    results = []
    try:
        print(f"Using model type: svm (standard for speech-music)")
        # mtFileClassification returns:
        # 1. segment_boundaries_seconds (list of segment start/end times)
        # 2. segment_labels_indices (list of class indices for each segment)
        # 3. class_names (list of class names, e.g., ['speech', 'music'])
        # The fourth argument is the model name (e.g., "svm_rbf_sm" without the path)
        # The fifth argument is usually 'svm' or 'knn' etc. for the model type.
        # Setting plot_results=False and verbose=False
        
        # mtFileClassification expects the model name, not the full path to the .model file
        # for the second argument if it's one of the library's models.
        # However, if it's a custom model, it expects the full path.
        # The pyAudioAnalysis documentation for mtFileClassification:
        # "model_name: path of the classification model"
        # "model_type: type of the classification model (svm, knn, randomforest, gradientboosting, extratrees)"
        # Let's stick to passing the full model_path for clarity.
        
        # Important: mtFileClassification also needs segment "step" and "window"
        # parameters, often passed as mt_step and mt_win.
        # Default values are often 1.0 sec for window and 0.5 sec for step.
        # These are not directly parameters to mtFileClassification but are often
        # assumed or must be part of the model's configuration if not specified.
        # The function signature is:
        # mtFileClassification(inputFile, model_name, model_type, plot_results=False, gt_file="", hmm_model_name="", segment_class_idx=-1, verbose=False)
        # It seems the function itself uses default mid-term window/step values (e.g. 1.0 sec window, 0.05 step)
        # and then it aggregates short-term classifications.

        # The model_name argument to mtFileClassification should be the path to the model file.
        # The model_type is 'svm' for svm_rbf_sm.
        segments_boundaries, class_indices, class_names = mtFileClassification(
            wav_file_path, model_path, "svm", plot_results=False, verbose=False
        )

        # Ensure class_names is what we expect, otherwise use default
        if not class_names or len(class_names) < 2:
            print(f"Warning: Model did not return expected class names. Got: {class_names}. Using default: {DEFAULT_CLASS_NAMES}")
            final_class_names = DEFAULT_CLASS_NAMES
        else:
            final_class_names = class_names
            print(f"Model returned class names: {final_class_names}")


        for i, seg_boundary in enumerate(segments_boundaries):
            start_time = seg_boundary[0]
            end_time = seg_boundary[1]
            label_index = int(class_indices[i]) # class_indices are usually float

            if label_index < len(final_class_names):
                label = final_class_names[label_index]
            else:
                label = "unknown"
                print(f"Warning: label_index {label_index} is out of bounds for class_names {final_class_names}")

            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "label": label
            })
        
        if not results:
            print("Classification completed but produced no segments.")

    except ImportError: # Should have been caught earlier, but as a safeguard
        print("Error: pyAudioAnalysis.audioSegmentation.mtFileClassification not available.")
        return [{"start_time": 0.0, "end_time": 0.0, "label": "classification_error (import)"}]
    except Exception as e:
        print(f"Error during pyAudioAnalysis classification for '{wav_file_path}': {e}")
        # Provide more context if possible
        if "model_name" in str(e) or "model_type" in str(e):
             print("This might be related to the model file path or type. Check model exists and path is correct.")
        return [{"start_time": 0.0, "end_time": 0.0, "label": f"classification_error ({type(e).__name__})"}]

    return results


def format_timecode(seconds_float: float) -> str:
    """
    Converts seconds (float) to a timecode string in HH:MM:SS.mmm format.
    """
    if seconds_float < 0:
        seconds_float = 0.0 # Ensure non-negative time
        
    # Create a timedelta object from the seconds
    td = timedelta(seconds=seconds_float)
    
    # Create a base datetime (midnight) and add the timedelta
    # This helps in formatting to HH:MM:SS
    base_dt = datetime.min # Represents 00:00:00
    dt_obj = base_dt + td
    
    # Format the time, and explicitly handle milliseconds
    # dt_obj.microsecond gives microseconds, so divide by 1000 for milliseconds
    milliseconds = dt_obj.microsecond // 1000
    formatted_time = dt_obj.strftime('%H:%M:%S') + f".{milliseconds:03d}"
    
    return formatted_time

def write_logic_pro_csv(segments: list, output_csv_path: str):
    """
    Writes audio segments to a CSV file formatted for Logic Pro.

    Args:
        segments: A list of segment dictionaries (from classify_audio_segments).
        output_csv_path: Path to save the output CSV file.
    """
    if not segments:
        print("No segments to write to CSV.")
        return

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header
            csv_writer.writerow(["Marker", "Position", "Name"])
            
            # Write segment data
            # Logic Pro uses "Marker" as a generic type for these text cues.
            # The 'Position' is the start time of the segment.
            # The 'Name' is the classification label.
            marker_type = "Marker" 
            for i, segment in enumerate(segments):
                start_time_float = segment.get("start_time", 0.0)
                label = segment.get("label", "Unknown")
                
                # Format timecode using the new function
                formatted_timecode = format_timecode(start_time_float)
                
                # Some DAWs might prefer marker names to be unique if used as actual markers.
                # Here, we're just using the label, but one could add numbers: f"{label}_{i+1}"
                csv_writer.writerow([marker_type, formatted_timecode, label])
        
        print(f"Successfully wrote Logic Pro CSV to: {output_csv_path}")
    except IOError as e:
        print(f"Error writing CSV file '{output_csv_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing CSV '{output_csv_path}': {e}")

def write_dcsx_json(input_filename: str, segments: list, output_dcsx_path: str):
    """
    Writes audio segments to a .dcsx JSON file.

    Args:
        input_filename: The name of the original audio file.
        segments: A list of segment dictionaries.
        output_dcsx_path: Path to save the output .dcsx JSON file.
    """
    if not segments:
        print("No segments to write to .dcsx JSON.")
        return

    data_structure = {
        "source_audio_file": os.path.basename(input_filename), # Store only the filename
        "cues": []
    }

    for i, segment in enumerate(segments):
        start_time_float = segment.get("start_time", 0.0)
        label = segment.get("label", "Unknown")
        
        cue = {
            "CueID": i + 1,
            "StartTimecode": format_timecode(start_time_float),
            "Label": label
            # Optionally, could add "EndTimecode": format_timecode(segment.get("end_time", 0.0))
        }
        data_structure["cues"].append(cue)

    try:
        with open(output_dcsx_path, 'w') as jsonfile:
            json.dump(data_structure, jsonfile, indent=4)
        print(f"Successfully wrote .dcsx JSON to: {output_dcsx_path}")
    except IOError as e:
        print(f"Error writing .dcsx JSON file '{output_dcsx_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing .dcsx JSON '{output_dcsx_path}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify audio segments from an input file and export results.")
    parser.add_argument(
        "input_audio_file",
        help="Path to the input audio file (e.g., WAV, MP3, MP4, MOV)."
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the pyAudioAnalysis SVM model file. Defaults to the model included with pyAudioAnalysis."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the output CSV and .dcsx files. Defaults to the input file's directory."
    )
    args = parser.parse_args()

    print("Starting audio classifier script...")

    input_file_path = args.input_audio_file
    model_to_use = args.model_path

    if not os.path.exists(input_file_path):
        print(f"Error: Input audio file not found: {input_file_path}")
        exit(1)

    # Determine output directory
    if args.output_dir:
        output_directory = args.output_dir
    else:
        output_directory = os.path.dirname(input_file_path)
        if not output_directory:  # If input file is in current dir, dirname is empty
            output_directory = "."

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_directory}': {e}")
        exit(1)
        
    # Define output filenames based on input filename
    base_filename_no_ext = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Path for the (potentially) converted WAV file
    # It's good practice to put intermediate files in the output_directory too.
    converted_wav_path = os.path.join(output_directory, base_filename_no_ext + "_converted.wav")
    
    output_csv_file = os.path.join(output_directory, base_filename_no_ext + "_logic_pro.csv")
    output_dcsx_file = os.path.join(output_directory, base_filename_no_ext + ".dcsx")

    print(f"Input audio: {input_file_path}")
    print(f"Model path: {model_to_use}")
    print(f"Output directory: {output_directory}")
    print(f"Target WAV output (if conversion needed): {converted_wav_path}")
    print(f"Output CSV: {output_csv_file}")
    print(f"Output .dcsx: {output_dcsx_file}")

    # 1. Convert to WAV
    wav_file = convert_to_wav(input_file_path, converted_wav_path)

    if wav_file:
        print(f"WAV file ready at: {wav_file}")
        # 2. Classify segments
        print(f"Using model path for classification: {model_to_use}")
        segments = classify_audio_segments(wav_file, model_path=model_to_use)
        
        if segments:
            print("\nClassification Results:")
            for seg in segments:
                print(f"  Start: {seg['start_time']:.2f}s, End: {seg['end_time']:.2f}s, Label: {seg['label']}")

            # Check if segments actually contain valid classification data, not error messages
            if not any("error" in seg["label"].lower() for seg in segments if isinstance(seg.get("label"), str)):
                 # 3. Write results to Logic Pro CSV
                 write_logic_pro_csv(segments, output_csv_file)
                 
                 # 4. Write results to .dcsx JSON
                 # Pass the original input file name for the 'source_audio_file' field in dcsx
                 write_dcsx_json(input_file_path, segments, output_dcsx_file)
            else:
                print("Skipping CSV and .dcsx export due to classification errors or empty results.")
        else:
            print("Classification returned no segments or failed.")
    else:
        print("Failed to convert audio to WAV. Cannot proceed with classification.")

    print("\nAudio classifier script finished.")
    print(f"Default model path (if not overridden by --model_path): {DEFAULT_MODEL_PATH}")
