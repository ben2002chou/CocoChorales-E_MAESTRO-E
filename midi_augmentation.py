import numpy as np
import pretty_midi
import argparse
import glob
import copy
import sys, os
from itertools import islice
from tqdm import tqdm
from utils.file_utils import get_config
from scipy.stats import truncnorm
from mistake_augmentations import augment_mistakes


from utils.file_utils import json_dump



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


def midi_augmentation(
    file_path, output_dir, config, generate_mistakes=False, overlap=True
):

    midi = pretty_midi.PrettyMIDI(file_path)

    midi_copy = copy.deepcopy(midi)

    if generate_mistakes:
        midi_modified, midi_extra_notes, midi_missed_notes, midi_correct_notes = augment_mistakes(
            midi_copy, overlap=overlap
        )

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
        "correct_notes": os.path.join(
            output_dir, "label", "correct_notes", base_dir, "stems_midi"
        ),
    }

    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # save original midi as mix.mid
    midi.write(os.path.join(paths["orginal_mix"], "mix.mid"))
    # save modified midi as mix.mid
    if generate_mistakes:
        midi_modified.write(os.path.join(paths["modified_mix"], "mix.mid"))

    for part_number, instrument in enumerate(midi.instruments):

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
                save_instrument_midi(
                    midi_correct_notes.instruments[part_number],
                    paths["correct_notes"],
                    part_number,
                )
            except IndexError:
                print(
                    f"No matching instrument for part_number {part_number} in modified/extra/missed notes MIDI."
                )
    if generate_mistakes:
        return (
            paths["original"],
            paths["modified"],
            paths["extra_notes"],
            paths["missed_notes"],
            paths["correct_notes"],
        )
    else:
        return paths["original"]

def augment_midi_files(
    midi_dir,
    output_dir,
    num_tracks,
    config=get_config(),
    expressive_timing=True,
    generate_mistakes=False,
    skip_existing_files=False,
    use_existing_audio=False,
    overlap=True,
):

    midi_file_list = glob.glob(f"{midi_dir}/**/*.mid", recursive=True) + glob.glob(
        f"{midi_dir}/**/*.midi", recursive=True
    )

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
            os.path.join(output_dir, "label", "correct_notes", base_dir, "stems_midi"),
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
            overlap=overlap,
        )

    paths = {
        "original": os.path.join(output_dir, "score"),
        "modified": os.path.join(output_dir, "mistake"),
        "extra_notes": os.path.join(output_dir, "label", "extra_notes"),
        "missed_notes": os.path.join(output_dir, "label", "removed_notes"),
        "correct_notes": os.path.join(output_dir, "label", "correct_notes"),
    }

    return (
        midi_file_list,
        paths["original"],
        paths["modified"],
        paths["extra_notes"],
        paths["missed_notes"],
        paths["correct_notes"],
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
        "--overlap",
        action="store_true",
        default=False,
        help="whether to allow overlapping notes in the MIDI files.",
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
        overlap=args.overlap,
    )
