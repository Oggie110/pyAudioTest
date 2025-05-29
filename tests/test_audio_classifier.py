import os
import subprocess
import unittest
import tempfile
import json

class TestAudioClassifier(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        # Assuming the script is run from the root of the project
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.script_path = os.path.join(self.base_dir, 'src', 'audio_classifier.py')
        self.sample_audio_path = os.path.join(self.base_dir, 'sample_audio', 'test_sample.wav')
        
        # Create a temporary directory for outputs
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir_obj.name

        # Check if the sample audio file exists
        if not os.path.exists(self.sample_audio_path):
            self.fail(f"Sample audio file not found at: {self.sample_audio_path}. "
                      "Please ensure it was downloaded or created in the previous step.")

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.temp_dir_obj.cleanup()

    def test_script_runs_and_creates_outputs(self):
        """Test if the audio_classifier.py script runs and creates expected output files."""
        
        command = [
            'python',
            self.script_path,
            self.sample_audio_path,
            '--output_dir', self.temp_dir_path
            # '--model_path', 'path/to/your/model' # Optional: if you want to test a specific model
        ]

        # Execute the script
        result = subprocess.run(command, capture_output=True, text=True)

        # Assert script execution was successful
        self.assertEqual(result.returncode, 0, 
                         f"Script execution failed with return code {result.returncode}.\n"
                         f"Stderr: {result.stderr}\nStdout: {result.stdout}")

        # Define expected output file paths
        # The script creates filenames based on the input audio filename
        base_input_filename = os.path.splitext(os.path.basename(self.sample_audio_path))[0]
        expected_csv_filename = base_input_filename + "_logic_pro.csv"
        expected_dcsx_filename = base_input_filename + ".dcsx"
        
        expected_csv_path = os.path.join(self.temp_dir_path, expected_csv_filename)
        expected_dcsx_path = os.path.join(self.temp_dir_path, expected_dcsx_filename)

        # Assert output files exist
        self.assertTrue(os.path.exists(expected_csv_path),
                        f"Expected CSV output file not found at: {expected_csv_path}\n"
                        f"Script stdout: {result.stdout}\nScript stderr: {result.stderr}")
        self.assertTrue(os.path.exists(expected_dcsx_path),
                        f"Expected .dcsx output file not found at: {expected_dcsx_path}\n"
                        f"Script stdout: {result.stdout}\nScript stderr: {result.stderr}")

        # Optional: Basic content check for CSV
        if os.path.exists(expected_csv_path):
            with open(expected_csv_path, 'r') as csv_file:
                lines = csv_file.readlines()
                self.assertTrue(len(lines) >= 1, # Header only is possible if no segments found
                                "CSV file should have at least a header line.")
                if len(lines) > 1: # If there are data rows
                    self.assertTrue("Marker,Position,Name" in lines[0], "CSV header is incorrect.")


        # Optional: Basic content check for .dcsx (ensure it's valid JSON)
        if os.path.exists(expected_dcsx_path):
            try:
                with open(expected_dcsx_path, 'r') as dcsx_file:
                    data = json.load(dcsx_file)
                self.assertIn("source_audio_file", data, ".dcsx missing 'source_audio_file' key.")
                self.assertIn("cues", data, ".dcsx missing 'cues' key.")
                self.assertIsInstance(data["cues"], list, "'cues' should be a list.")
                if data["cues"]: # If there are cues, check the first one
                    first_cue = data["cues"][0]
                    self.assertIn("CueID", first_cue)
                    self.assertIn("StartTimecode", first_cue)
                    self.assertIn("Label", first_cue)

            except json.JSONDecodeError:
                self.fail(f".dcsx file is not valid JSON: {expected_dcsx_path}")
            except Exception as e:
                self.fail(f"Error during .dcsx content check: {e}")

if __name__ == '__main__':
    unittest.main()
