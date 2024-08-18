import librosa
import soundfile as sf
import os
import subprocess
import numpy as np

def remove_silence_and_save(audio_path, output_path, top_db=40):
    y, sr = librosa.load(audio_path, sr=None)
    yt, index = librosa.effects.trim(y, top_db=top_db)
    initial_silence_duration = index[0] / sr
    sf.write(output_path, yt, sr)
    print(f"Sample rate: {sr}, Trimmed length: {len(yt) / sr:.2f} seconds")
    return len(yt) / sr, initial_silence_duration

def time_stretch(input_file, output_file, factor):
    cmd = f"ffmpeg -i {input_file} -filter:a 'atempo={factor}' {output_file} -y"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    print("FFmpeg Output:", result.stdout)
    print("FFmpeg Error:", result.stderr)

def get_audio_length(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return len(y) / sr

def add_silence(audio_data, sr, silence_duration):
    # Calculate the number of silent samples to add
    silent_samples = int(silence_duration * sr)
    # Create an array of zeros (silent samples)
    silence = np.zeros(silent_samples)
    # Concatenate silence at the beginning of the audio data
    return np.concatenate((silence, audio_data))

def process_directory(base_path):
    score_path = os.path.join(base_path, 'score')
    mistake_path = os.path.join(base_path, 'mistake')

    for track_dir in os.listdir(score_path):
        track_score_path = os.path.join(score_path, track_dir)
        track_mistake_path = os.path.join(mistake_path, track_dir)

        if os.path.exists(track_score_path) and os.path.exists(track_mistake_path):
            mix_file = 'mix.wav'
            score_audio_path = os.path.join(track_score_path, mix_file)
            mistake_audio_path = os.path.join(track_mistake_path, mix_file)

            if os.path.isfile(score_audio_path) and os.path.isfile(mistake_audio_path):
                trimmed_score_path = os.path.join(track_score_path, f"trimmed_{mix_file}")
                trimmed_mistake_path = os.path.join(track_mistake_path, f"trimmed_{mix_file}")

                new_length_score, _ = remove_silence_and_save(score_audio_path, trimmed_score_path)
                new_length_mistake, mistake_initial_silence = remove_silence_and_save(mistake_audio_path, trimmed_mistake_path)

                if new_length_mistake != 0:
                    stretch_factor = new_length_mistake / new_length_score
                else:
                    stretch_factor = 1

                stretched_score_path = os.path.join(track_score_path, f"stretched_{mix_file}")
                time_stretch(trimmed_score_path, stretched_score_path, 1 / stretch_factor)

                # Load stretched audio and add back the original silence
                y, sr = librosa.load(stretched_score_path, sr=None)
                y_with_silence = add_silence(y, sr, mistake_initial_silence)
                print(f"initial_silence: {mistake_initial_silence:.2f} seconds")
                final_path = os.path.join(track_score_path, mix_file)  # Replace original mix.wav
                sf.write(final_path, y_with_silence, sr)

                print(f"Processed {track_dir}:")
                print(f"  New Length of Stretched Score File with Silence: {len(y_with_silence) / sr:.2f} seconds")
            else:
                print(f"Skipping {track_dir}, mix.wav not found in both directories.")
        else:
            print(f"Skipping {track_dir}, corresponding directories do not exist.")

# Example usage
base_path = '/home/chou150/depot/datasets/Score_Informed_with_mistakes_stretched'
process_directory(base_path)
