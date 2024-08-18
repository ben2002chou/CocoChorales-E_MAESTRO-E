import fluidsynth

# Path to your SoundFont and MIDI file
sf2_path = (
    "/home/chou150/code/chamber-ensemble-generator/sf2/YDP-GrandPiano-20160804.sf2"
)
midi_file = "/home/chou150/depot/datasets/maestro/maestro_with_mistakes/mistake/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_04_Track04_wav/mix.mid"
audio_output = "/home/chou150/code/chamber-ensemble-generator/output/output.wav"

# Initialize FluidSynth
synth = fluidsynth.Synth()

# Specify the audio driver to avoid ALSA, use 'file' for direct output
synth.start(driver="file", file=audio_output)

# Load SoundFont
sfid = synth.sfload(sf2_path)
synth.program_select(0, sfid, 0, 0)

# If midi_file_play is unavailable, you may need to find another method or script to play MIDI files
# Possible alternative: Load and play MIDI data manually or update/use another library

# Cleanup
synth.delete()
fluidsynth -F /home/chou150/code/chamber-ensemble-generator/output/output.wav -T wav -ni /home/chou150/code/chamber-ensemble-generator/sf2/YDP-GrandPiano-20160804.sf2 /home/chou150/depot/datasets/maestro/maestro_with_mistakes/score/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav/mix.mid

fluidsynth -F /home/chou150/code/chamber-ensemble-generator/output/output.wav -ni /home/chou150/code/chamber-ensemble-generator/sf2/YDP-GrandPiano-20160804.sf2 /home/chou150/depot/datasets/maestro/maestro_with_mistakes/mistake/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_04_Track04_wav/mix.mid