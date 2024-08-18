import numpy as np
from mido import MidiFile
import pretty_midi
import copy
import argparse
import glob
import sys, os
from itertools import islice
from tqdm import tqdm
from utils.file_utils import get_config
from scipy.stats import truncnorm
from mistake_augmentations import augment_mistakes
from mistake_augmentations import add_screwups, add_pitch_bends

from utils.file_utils import json_dump
import wandb


def make_instrument_mono(instrument):
    """
    Make the instrument monophonic by adjusting the note durations.

    Args:
        instrument (Instrument): The instrument to make monophonic.

    Returns:
        None
    """
    all_notes = instrument.notes
    for i in range(len(all_notes)):
        if i != len(all_notes) - 1:
            if all_notes[i].end > all_notes[i + 1].start:
                all_notes[i].end = all_notes[i + 1].start
    instrument.notes = all_notes


def save_instrument_midi(instrument, path, part_number):
    instrument_name = (
        pretty_midi.program_to_instrument_name(instrument.program)
        .replace(" ", "_")
        .lower()
    )
    # Utility function to save an instrument's MIDI data to a specified path
    midi_data = pretty_midi.PrettyMIDI()
    midi_data.instruments.append(instrument)
    midi_file_path = os.path.join(path, f"{part_number}_{instrument_name}.mid")
    midi_data.write(midi_file_path)


def midi_augmentation(file_path, output_dir, config, generate_mistakes=False):

    midi = pretty_midi.PrettyMIDI(file_path)

    midi_copy = copy.deepcopy(midi)

    if generate_mistakes:
        midi_modified, midi_extra_notes, midi_missed_notes = augment_mistakes(midi_copy)

    base_dir = os.path.splitext(os.path.basename(file_path))[0]
    paths = {
        "orginal_mix": os.path.join(output_dir, "score", base_dir),
        "original": os.path.join(output_dir, "score", base_dir, "stems_midi"),
        "modified_mix": os.path.join(output_dir, "mistake", base_dir),
        "modified": os.path.join(output_dir, "mistake", base_dir, "stems_midi"),
        "extra_notes": os.path.join(
            output_dir, "label", "extra_notes", base_dir, "stems_midi"
        ),
        "missed_notes": os.path.join(
            output_dir, "label", "removed_notes", base_dir, "stems_midi"
        ),
    }

    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # save original midi as mix.mid
    midi.write(os.path.join(paths["orginal_mix"], "mix.mid"))
    # save modified midi as mix.mid
    midi_modified.write(os.path.join(paths["modified_mix"], "mix.mid"))

    for part_number, instrument in enumerate(midi.instruments):
        if "piano" in instrument.name.lower():
            overlap = True
        else:
            overlap = False
        # Make the instrument monophonic for midi-ddsp
        if overlap is False:
            make_instrument_mono(instrument)
            make_instrument_mono(midi_modified.instruments[part_number])
            make_instrument_mono(midi_extra_notes.instruments[part_number])
            make_instrument_mono(midi_missed_notes.instruments[part_number])

        # Save original instrument MIDI
        save_instrument_midi(instrument, paths["original"], part_number)

        # Save modified instrument MIDI
        if generate_mistakes:
            try:
                save_instrument_midi(
                    midi_modified.instruments[part_number],
                    paths["modified"],
                    part_number,
                )
                save_instrument_midi(
                    midi_extra_notes.instruments[part_number],
                    paths["extra_notes"],
                    part_number,
                )
                save_instrument_midi(
                    midi_missed_notes.instruments[part_number],
                    paths["missed_notes"],
                    part_number,
                )
            except IndexError:
                print(
                    f"No matching instrument for part_number {part_number} in modified/extra/missed notes MIDI."
                )

    return (
        paths["original"],
        paths["modified"],
        paths["extra_notes"],
        paths["missed_notes"],
    )


def augment_midi_files(
    midi_dir,
    output_dir,
    num_tracks,
    config=get_config(),
    expressive_timing=True,
    generate_mistakes=False,
    skip_existing_files=False,
):

    midi_file_list = glob.glob(f"{midi_dir}/**/*.mid", recursive=True)

    os.makedirs(output_dir, exist_ok=True)

    for midi_file in tqdm(
        islice(midi_file_list, num_tracks), desc="Augmenting MIDI files"
    ):
        base_dir = os.path.splitext(os.path.basename(midi_file))[0]
        skip_file = True

        # Define potential output paths
        paths_to_check = [
            os.path.join(output_dir, "score", base_dir, "mix.mid"),
            os.path.join(output_dir, "score", base_dir, "stems_midi"),
            os.path.join(output_dir, "mistake", base_dir, "mix.mid"),
            os.path.join(output_dir, "mistake", base_dir, "stems_midi"),
            os.path.join(output_dir, "label", "extra_notes", base_dir, "stems_midi"),
            os.path.join(output_dir, "label", "removed_notes", base_dir, "stems_midi"),
        ]
        print(paths_to_check)

        if skip_existing_files:
            # Check if all the files for the current MIDI file already exist
            for path in paths_to_check:
                if not os.path.exists(path):
                    # If any of the directories do not exist, we need to process this file
                    skip_file = False
                    break

            if skip_file:
                print(f"Skipping {midi_file} as augmented files already exist.")
                continue

        midi_augmentation(
            midi_file,
            output_dir,
            config,
            generate_mistakes=generate_mistakes,
        )

    paths = {
        "original": os.path.join(output_dir, "score"),
        "modified": os.path.join(output_dir, "mistake"),
        "extra_notes": os.path.join(output_dir, "label", "extra_notes"),
        "missed_notes": os.path.join(output_dir, "label", "removed_notes"),
    }

    return (
        midi_file_list,
        paths["original"],
        paths["modified"],
        paths["extra_notes"],
        paths["missed_notes"],
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MIDI augmentation")
    parser.add_argument(
        "--midi_dir",
        type=str,
        required=True,
        help="the directory containing all the MIDI files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the directory for outputting the augmented MIDI files.",
    )
    parser.add_argument(
        "--generate_mistakes",
        action="store_true",
        help="whether to generate errors in the MIDI files.",
    )
    parser.add_argument(
        "--num_tracks",
        type=int,
        default=1000,
        help="the number of tracks to generate.",
    )
    args = parser.parse_args()

    config = get_config()

    augment_midi_files(
        args.midi_dir,
        args.output_dir,
        args.num_tracks,
        config,
        generate_mistakes=args.generate_mistakes,
    )
