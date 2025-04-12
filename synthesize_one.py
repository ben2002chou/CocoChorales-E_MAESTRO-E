import argparse
from midi_ddsp_synthesize import synthesize_midi, load_pretrained_model


def synthesize_single_midi(
    midi_path,
    output_dir,
    synthesis_generator_weight_path,
    expression_generator_weight_path,
    pitch_offset=0,
    speed_rate=1.0,
    sf2_path="/home/chou150/code/chamber-ensemble-generator/sf2/YDP-GrandPiano-20160804.sf2",
    use_fluidsynth=False,
    skip_existing_files=True,
    save_metadata=False,
):
    """
    Synthesize a single MIDI file using MIDI-DDSP with the default parameters.
    """
    try:
        # Load pre-trained models
        synthesis_generator, expression_generator = load_pretrained_model(
            synthesis_generator_path=synthesis_generator_weight_path,
            expression_generator_path=expression_generator_weight_path,
        )

        # Synthesize the MIDI file
        synthesize_midi(
            synthesis_generator,
            expression_generator,
            midi_path,
            pitch_offset=pitch_offset,
            speed_rate=speed_rate,
            output_dir=output_dir,
            sf2_path=sf2_path,
            use_fluidsynth=use_fluidsynth,
            display_progressbar=False,
            skip_existing_files=skip_existing_files,
            save_metadata=save_metadata,
        )
        print(f"Synthesized {midi_path} and saved to {output_dir}")
    except Exception as e:
        print(f"Error synthesizing {midi_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize a single MIDI file.")
    parser.add_argument(
        "--midi_path", required=True, help="Path to the MIDI file to synthesize."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the synthesized audio."
    )
    parser.add_argument(
        "--synthesis_generator_weight_path",
        type=str,
        default=None,
        help="Path to the Synthesis Generator weights.",
    )
    parser.add_argument(
        "--expression_generator_weight_path",
        type=str,
        default=None,
        help="Path to the Expression Generator weights.",
    )
    parser.add_argument(
        "--pitch_offset",
        type=int,
        default=0,
        help="Pitch offset to transpose the MIDI in semitone.",
    )
    parser.add_argument(
        "--speed_rate",
        type=float,
        default=48/59,
        help="Speed rate for synthesizing the MIDI file.",
    )
    parser.add_argument(
        "--sf2_path",
        default="/home/chou150/code/chamber-ensemble-generator/sf2/YDP-GrandPiano-20160804.sf2",
        help="Path to a SF2 soundfont file.",
    )
    parser.add_argument(
        "--use_fluidsynth",
        action="store_true",
        help="Use FluidSynth for synthesizing instruments not in MIDI-DDSP.",
    )
    parser.add_argument(
        "--skip_existing_files",
        action="store_true",
        help="Skip synthesizing if the output already exists.",
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save metadata including instrument_id and synthesis parameters.",
    )

    args = parser.parse_args()

    synthesize_single_midi(
        midi_path=args.midi_path,
        output_dir=args.output_dir,
        synthesis_generator_weight_path=args.synthesis_generator_weight_path,
        expression_generator_weight_path=args.expression_generator_weight_path,
        pitch_offset=args.pitch_offset,
        speed_rate=args.speed_rate,
        sf2_path=args.sf2_path,
        use_fluidsynth=args.use_fluidsynth,
        skip_existing_files=args.skip_existing_files,
        save_metadata=args.save_metadata,
    )
# python synthesize_one.py --midi_path "/home/chou150/code/Muse/physical_test_data/real/cello/demo_2_score.mid"  --output_dir "/home/chou150/code/Muse/physical_test_data/real/cello/"
# python synthesize_one.py --midi_path "/home/chou150/code/Muse/physical_test_data/Clarinet_mistake.mid"  --output_dir "/home/chou150/code/Muse/physical_test_data"