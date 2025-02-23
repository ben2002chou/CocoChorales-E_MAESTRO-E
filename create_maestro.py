import sys
import glob
import os
import argparse
import numpy as np
import shutil
import traceback
from tqdm import tqdm  # Add missing import statement

import multiprocessing
import midi_augmentation
from midi_ddsp_synthesize import synthesize_midi, load_pretrained_model


from utils.file_utils import get_config


from functools import partial


def check_and_copy_existing_audio(midi_file, score_dir, use_existing_audio):
    if not use_existing_audio:
        return False

    # Extract the base name without the .mid or .midi extension
    base_name = os.path.splitext(os.path.basename(midi_file))[0]

    # Construct the expected path for the mix.mid and mix.wav in the score directory
    expected_mix_mid_path = os.path.join(score_dir, base_name, "mix.mid")
    expected_mix_wav_path = os.path.join(score_dir, base_name, "mix.wav")

    # Check if a .wav file with the same base name exists in the same directory as the MIDI file
    source_wav_file = os.path.join(os.path.dirname(midi_file), base_name + ".wav")
    print(f"Checking for existing audio file: {source_wav_file}")

    if os.path.exists(source_wav_file):
        # Check if the .wav file needs to be copied or already exists at the target location
        if not os.path.exists(expected_mix_wav_path):
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(expected_mix_wav_path), exist_ok=True)

            shutil.copy(source_wav_file, expected_mix_wav_path)
            print(f"Copied existing audio for {midi_file} to {expected_mix_wav_path}")
        else:
            print(
                f"Existing audio already present for {midi_file} at {expected_mix_wav_path}"
            )
        return True
    else:
        print(f"No existing audio file found for {midi_file}.")
        return False


def parse_synthesis_midi_files(
    midi_dir=None,
    pitch_offset=0,
    speed_rate=1.0,
    sf2_path=None,
    use_fluidsynth=False,
    synthesis_generator_weight_path=None,
    expression_generator_weight_path=None,
    skip_existing_files=True,
    save_metadata=True,
):
    """
    Synthesize MIDI files using MIDI-DDSP with specified parameters.

    :param midi_dir: Directory containing MIDI files.
    :param pitch_offset: Pitch offset to transpose in semitone.
    :param speed_rate: The speed to synthesize the MIDI file(s).
    :param sf2_path: Path to a sf2 soundfont file.
    :param use_fluidsynth: Use FluidSynth for synthesizing midi instruments not contained in MIDI-DDSP.
    :param synthesis_generator_weight_path: Path to the Synthesis Generator weights.
    :param expression_generator_weight_path: Path to the expression generator weights.
    :param skip_existing_files: Skip synthesizing MIDI files if they already exist in the output folder.
    :param save_metadata: Save metadata including instrument_id, note expression controls, and synthesis parameters.
    """

    return (
        pitch_offset,
        speed_rate,
        sf2_path,
        use_fluidsynth,
        synthesis_generator_weight_path,
        expression_generator_weight_path,
        skip_existing_files,
        save_metadata,
    )


# Wrapper function for multiprocessing
def process_midi_file_wrapper(
    midi_file,
    synthesis_generator_weight_path,
    expression_generator_weight_path,
    pitch_offset,
    speed_rate,
    sf2_path,
    use_fluidsynth,
    skip_existing_files,
    save_metadata,
):
    # print(f"Processing {midi_file}", flush=True)
    # Split the path into parts
    path_parts = midi_file.split(os.sep)

    # Remove 'stems_midi' from the list of parts
    path_parts = [part for part in path_parts if part != "stems_midi"]

    # Reconstruct the path without 'stems_midi'
    custom_output_dir = os.sep.join(path_parts)

    # Extract directory path without the file name
    custom_output_dir = os.path.dirname(custom_output_dir)

    # Ensure the directory exists
    os.makedirs(custom_output_dir, exist_ok=True)
    try:
        synthesis_generator, expression_generator = load_pretrained_model(
            synthesis_generator_path=synthesis_generator_weight_path,
            expression_generator_path=expression_generator_weight_path,
        )
        # print(f"Synthesizing {midi_file}", flush=True)
        print("skip_existing_files: ", skip_existing_files)
        synthesize_midi(
            synthesis_generator,
            expression_generator,
            midi_file,
            pitch_offset=pitch_offset,
            speed_rate=speed_rate,
            output_dir=custom_output_dir,
            sf2_path=sf2_path,
            use_fluidsynth=use_fluidsynth,
            display_progressbar=False,
            skip_existing_files=skip_existing_files,
            save_metadata=save_metadata,
        )
        print(
            f"Finished synthesizing {midi_file} saving to {custom_output_dir}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()  # This prints the full stack trace
        print(f"An error occurred: {str(e)}")  # Printing the exception message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MIDI files for ensemble generation."
    )
    parser.add_argument(
        "--code_base",
        default="/home/chou150/depot/code/chamber-ensemble-generator/",
        help="Base path to the code repository.",
    )
    parser.add_argument(
        "--midi_dir",
        default="/home/chou150/depot/datasets/maestro/maestro-v3.0.0",
        help="Directory containing original MIDI files for augmentation.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/chou150/depot/datasets/maestro/maestro_with_mistakes",
        help="Directory to save augmented MIDI files.",
    )
    parser.add_argument(
        "--num_tracks", type=int, default=60000, help="Number of tracks to generate."
    )
    parser.add_argument(
        "--skip_existing_files",
        action="store_true",
        default=True,
        help="Skip processing of existing files.",
    )
    parser.add_argument(
        "--use_existing_audio",
        action="store_true",
        default=True,
        help="Use existing audio files if available.",
    )
    parser.add_argument(
        "--use_fluidsynth",
        action="store_true",
        default=True,
        help="Use FluidSynth for synthesizing MIDI instruments not in MIDI-DDSP.",
    )

    parser.add_argument(
        "--serial_processing",
        action="store_true",
        default=False,
        help="Process files in serial.",
    )
    
    parser.add_argument(
        "generate_mistakes",
        action="store_true",
        default=True,
        help="Generate mistakes in the MIDI files.",
    )    

    parser.add_argument(
        "--sf2_path",
        default="/home/chou150/code/chamber-ensemble-generator/sf2/YDP-GrandPiano-20160804.sf2",
        help="Path to the sf2 soundfont file.",
    )    
    args = parser.parse_args()

    sys.path.append(args.code_base)

    config = get_config()

    # MIDI Augmentation - This step is preparing the MIDI files for synthesis
    ########################################################################### (are we outputting the dir or individual files paths?)

    midi_file_list, score_dir, mistake_dir, extra_dir, removed_dir, correct_dir = (
        midi_augmentation.augment_midi_files(
            midi_dir=args.midi_dir,
            output_dir=args.output_dir,
            num_tracks=args.num_tracks,  # 60000
            generate_mistakes=args.generate_mistakes,
            skip_existing_files=args.skip_existing_files,
            use_existing_audio=args.use_existing_audio,
            overlap=True,
        )
    )
    (
        pitch_offset,
        speed_rate,
        sf2_path,
        use_fluidsynth,
        synthesis_generator_weight_path,
        expression_generator_weight_path,
        skip_existing_files,
        save_metadata,
    ) = parse_synthesis_midi_files(
        skip_existing_files=args.skip_existing_files,
        save_metadata=False,
        use_fluidsynth=args.use_fluidsynth,
        sf2_path=args.sf2_path,
    )

    process_midi_file_with_context = partial(
        process_midi_file_wrapper,
        synthesis_generator_weight_path=synthesis_generator_weight_path,
        expression_generator_weight_path=expression_generator_weight_path,
        pitch_offset=pitch_offset,
        speed_rate=speed_rate,
        sf2_path=sf2_path,
        use_fluidsynth=use_fluidsynth,
        skip_existing_files=skip_existing_files,
        save_metadata=save_metadata,
    )
    score_file_list = []
    modified_file_list = []

    for midi_file_path in midi_file_list:
        # Extract the base name without the .mid extension
        base_name = os.path.splitext(os.path.basename(midi_file_path))[0]

        # Construct the expected path to mix.mid in the corresponding directory under score_dir
        expected_mix_mid_path = os.path.join(score_dir, base_name, "mix.mid")

        # Check if the mix.mid file exists at the expected path
        if os.path.isfile(expected_mix_mid_path):
            if check_and_copy_existing_audio(
                midi_file_path, score_dir, args.use_existing_audio
            ):
                pass
            else:
                score_file_list.append(expected_mix_mid_path)
        else:
            print(f"Expected mix.mid file not found: {expected_mix_mid_path}")

        expected_modified_mid_path = os.path.join(mistake_dir, base_name, "mix.mid")
        if os.path.isfile(expected_modified_mid_path):
            modified_file_list.append(expected_modified_mid_path)
        else:
            print(f"Expected mix.mid file not found: {expected_modified_mid_path}")

    print(len(midi_file_list))
    print(len(score_file_list))
    print(len(modified_file_list))

    # Process the collected mix.mid files
    if score_file_list:

        if len(score_file_list) == 0:
            raise FileNotFoundError("No MIDI files found in the directory.")
        if args.serial_processing:
            print(f"Processing {len(score_file_list)} score files in serial")
            for midi_file in tqdm(score_file_list, desc="Processing score files"):
                process_midi_file_with_context(midi_file)
        else:
            num_cpus = int(
                os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())
            )
            num_processes = num_cpus
            print(f"Using {num_processes} processes for parallel processing")
            with multiprocessing.Pool(processes=num_processes) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(
                            process_midi_file_with_context, score_file_list
                        ),
                        total=len(score_file_list),
                        desc="Generating files: ",
                    )
                )

    # Repeat the process for the mistake directory in parallel
    if modified_file_list:

        if len(modified_file_list) == 0:
            raise FileNotFoundError("No MIDI files found in the directory.")
        if args.serial_processing:
            print(f"Processing {len(modified_file_list)} modified files in serial")
            for midi_file in tqdm(modified_file_list, desc="Processing modified files"):
                process_midi_file_with_context(midi_file)
        else:
            num_cpus = int(
                os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())
            )

            num_processes = num_cpus
            print(f"Using {num_processes} processes for parallel processing")
            with multiprocessing.Pool(processes=num_processes) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(
                            process_midi_file_with_context, modified_file_list
                        ),
                        total=len(modified_file_list),
                        desc="Generating files: ",
                    )
                )
