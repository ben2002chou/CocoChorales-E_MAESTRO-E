import os
import pretty_midi

def list_midi_instruments(directory):
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a MIDI file
        if filename.endswith(".mid") or filename.endswith(".midi"):
            filepath = os.path.join(directory, filename)
            try:
                # Load the MIDI file
                midi_data = pretty_midi.PrettyMIDI(filepath)
                
                # Print the name of the file
                print(f"Instruments in {filename}:")
                
                # List all instruments in the MIDI file
                for instrument in midi_data.instruments:
                    print(f" - {instrument.name} (Program: {instrument.program}, Is Drum: {instrument.is_drum})")
                
                print()  # Print a blank line for better readability
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Specify the path to the subfolder containing the MIDI files
midi_directory = '/home/chou150/depot/datasets/cocochorales_full/org_chunked_midi/brass/0'

# Call the function to list instruments
list_midi_instruments(midi_directory)