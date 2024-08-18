import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_peak_db_per_frame(audio_path, output_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)  # Use the native sampling rate
    # Compute the STFT
    D = librosa.stft(y)
    # Convert to decibel
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # Get the peak dB per frame
    peak_db_per_frame = np.max(S_db, axis=0)  # Taking the max in each column (frame)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(peak_db_per_frame)
    plt.title('Peak dB level per frame')
    plt.xlabel('Frame')
    plt.ylabel('dB')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def process_directory(directory, plot_directory):
    # Check if plot_directory exists, if not, create it
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    
    # List all .wav files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio_path = os.path.join(directory, filename)
            plot_filename = filename.replace('.wav', '_peak_db_per_frame.png')
            output_path = os.path.join(plot_directory, plot_filename)
            plot_peak_db_per_frame(audio_path, output_path)
            print(f"Saved peak dB plot for {filename} as {output_path}")

# Usage: Replace 'path_to_your_audio_directory' and 'path_to_your_plot_directory'
process_directory('/home/chou150/depot/datasets/ScoreInformedPianoTranscriptionDataset', '/home/chou150/depot/datasets/ScoreInformedPianoTranscriptionDataset/plots/db')


