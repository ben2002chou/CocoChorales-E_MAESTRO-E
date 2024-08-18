import sys, os
import shutil

sys.path.append("/home/chou150/code/chamber-ensemble-generator/")

import mistake_augmentations
from mistake_augmentations import add_screwups, add_pitch_bends
import pretty_midi
import numpy as np


def generate_mistakes(midi_path, output_dir):
    midi = pretty_midi.PrettyMIDI(midi_path)
    pb = False  # TODO: pitch bends mistakes should be gotten from the actual complete midi file
    overlap = False # False
    # for inst in midi.instruments:
    #     if "piano" in inst.name.lower():
    #         pb = False
    #         overlap = True
    midi, midi_extra_notes, midi_removed_notes, midi_correct_notes = add_screwups(
        midi=midi,
        lambda_occur=0.5,  # 0.03
        stdev_pitch_delta=1,
        mean_duration=1,
        stdev_duration=0.02,
        allow_overlap=overlap,
        fixed_screwup_type=None,
    )
    if pb:
        add_pitch_bends(
            midi=midi,
            lambda_occur=2,
            mean_delta=0,
            stdev_delta=np.sqrt(1000),
            step_size=0.01,
        )
    # Remove the '.mid' extension and append '_modified.midi'
    base_path = output_dir 
    modified_midi_path = (
        base_path + "_modified.mid"
    )  # Adds the new part of the filename
    print(modified_midi_path)
    midi.write(modified_midi_path)
    extra_midi_path = base_path + "_extra.mid"
    print(extra_midi_path)
    midi_extra_notes.write(extra_midi_path)
    removed_midi_path = base_path + "_removed.mid"
    print(removed_midi_path)
    midi_removed_notes.write(removed_midi_path)
    correct_midi_path = base_path + "_correct.mid"
    print(correct_midi_path)
    midi_correct_notes.write(correct_midi_path)


if __name__ == "__main__":
    # path = input("Enter a path:\n")
    path = "/depot/yunglu/data/datasets_ben/cocochorales_full/main_dataset/test/string_track218400/stems_midi/4_cello.mid"
    output_dir = "/home/chou150/code/chamber-ensemble-generator/output/"
    generate_mistakes(path, output_dir)
    # save a copy of the original midi file
    shutil.copy(path, output_dir + "original.mid")
