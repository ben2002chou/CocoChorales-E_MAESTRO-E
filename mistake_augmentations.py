import math
import copy
import pretty_midi as pm
import numpy as np
from scipy.stats import truncnorm
from utils.file_utils import get_config

rng = np.random.default_rng()
""" 
Usage:
add_pitch_bends(midi=midi, lambda_occur=2, mean_delta=0, stdev_delta=np.sqrt(1000), step_size=0.01)
add_screwups(midi=midi, lambda_occur=0.03, stdev_pitch_delta=1)
"""


# def make_instrument_mono(instrument):
#     """
#     Make the instrument monophonic by adjusting the note durations.

#     Args:
#         instrument (Instrument): The instrument to make monophonic.

#     Returns:
#         None
#     """
#     all_notes = instrument.notes
#     all_notes.sort(key=(lambda x: x.start))
#     for i in range(len(all_notes)):
#         if i != len(all_notes) - 1:
#             if all_notes[i].end > all_notes[i + 1].start:
#                 all_notes[i].end = all_notes[i + 1].start
#     instrument.notes = all_notes


def notes_are_equal(note1, note2):
    """
    Determine if two pretty_midi.Note objects are equal based on start time, end time, and pitch.

    Args:
    note1 (pretty_midi.Note): The first note to compare.
    note2 (pretty_midi.Note): The second note to compare.

    Returns:
    bool: True if the notes are equal, False otherwise.
    """
    return (
        note1.start == note2.start
        and note1.end == note2.end
        and note1.pitch == note2.pitch
    )


# Not used for piano due to the piano having "keys" instead of "strings"
# TODO: check the usablity of this function
# TODO: use pretty_midi.pitch_bend_to_semitones(pitch_bend, semitone_range=2.0) to convert pitch bend to semitones
# def add_pitch_bends(midi, lambda_occur, mean_delta, stdev_delta, step_size):
#     for inst in midi.instruments:
#         inst.pitch_bends = []
#         # Flatten note times list
#         single_notes = []
#         last_time = 0.0
#         for note in inst.notes:
#             if note.start >= last_time:
#                 single_notes.append(note)
#                 last_time = note.end

#         fixed_bend_points = []
#         # Add fixed point pitch bends
#         for note in single_notes:
#             # Do 1 pitch bend at start of each note
#             # Define the boundaries of the truncated normal distribution
#             clip_a, clip_b = -8192, 8191

#             # Calculate the parameters for the truncated normal distribution
#             a, b = (clip_a - mean_delta) / (stdev_delta * stdev_delta), (
#                 clip_b - mean_delta
#             ) / (stdev_delta * stdev_delta)

#             # Generate a random number from the truncated normal distribution and scale it
#             bend = int(truncnorm(a, b).rvs() * (stdev_delta * stdev_delta) + mean_delta)

#             # The generated number is guaranteed to be within the range [-8192, 8191] due to the truncation
#             fixed_bend_points.append(pm.PitchBend(bend, note.start))
#             # Add more randomly
#             occurrences = rng.poisson(lam=lambda_occur * (note.end - note.start))
#             for i in range(occurrences):
#                 time = rng.uniform(low=note.start, high=note.end)
#                 bend = int(
#                     truncnorm(a, b).rvs() * (stdev_delta * stdev_delta) + mean_delta
#                 )
#                 fixed_bend_points.append(pm.PitchBend(bend, time))

#         # Sort by time from least to greatest
#         fixed_bend_points.sort(key=(lambda x: x.time))

#         # Linear interpolation
#         inst.pitch_bends = fixed_bend_points.copy()
#         for i in range(len(fixed_bend_points) - 1):
#             left_pitch_bend = fixed_bend_points[i]
#             right_pitch_bend = fixed_bend_points[i + 1]
#             n_points_to_add = int(
#                 np.floor((right_pitch_bend.time - left_pitch_bend.time) / step_size)
#             )
#             for j in range(1, n_points_to_add + 1):
#                 time = left_pitch_bend.time + j * step_size
#                 bend = int(
#                     left_pitch_bend.pitch
#                     + (
#                         (right_pitch_bend.pitch - left_pitch_bend.pitch)
#                         * (j / (n_points_to_add + 1))
#                     )
#                 )
#                 p = 1 - (j / (n_points_to_add + 1)) / 1.3
#                 # print(p)
#                 bend = int(math.copysign(1, bend) * pow(abs(bend), p))
#                 if bend > 8191:
#                     bend = 8191
#                 elif bend < -8192:
#                     bend = -8192
#                 inst.pitch_bends.append(pm.PitchBend(bend, time))


def get_thirty_second_note_duration(tempo, midi):
    """
    Calculate the duration of a thirty-second-note given the tempo and MIDI resolution.
    """
    ticks_per_beat = midi.resolution
    beats_per_second = tempo / 60.0
    ticks_per_second = ticks_per_beat * beats_per_second
    thirty_second_duration_seconds = (
        1 / 8
    ) / beats_per_second  # Quarter note divided by 4
    return thirty_second_duration_seconds


def adjust_for_overlap(inst, idx, duration, start_time, notes_to_remove, notes_to_add):
    # sort notes by start time
    if idx < len(inst.notes) - 1:
        next_note = inst.notes[idx + 1]
        if start_time + duration > next_note.start:

            # Adjust current note to avoid overlap
            # TODO: this could be problematic because the next note could be very far away?
            duration = next_note.start - start_time
            if duration <= 0:
                print(f"duration4: {duration}", flush=True)
                notes_to_remove.append(inst.notes[idx])
    if idx > 0:
        previous_note = inst.notes[idx - 1]
        if start_time < previous_note.end:
            # Adjust previous note to avoid overlap
            if start_time <= previous_note.start:
                notes_to_remove.append(previous_note)
                return duration, start_time, notes_to_remove, notes_to_add
                print("note start time is less than previous note start time")
            #     start_time = previous_note.start + 0.01
            #     end_time = start_time + duration
            # If previous note is a candidate for removal, adjust
            for note in notes_to_remove:
                if (
                    note.pitch == previous_note.pitch
                    and note.start == previous_note.start
                    and note.end == previous_note.end
                ):
                    note.end = min(note.end, start_time)
            # If previous note is a candidate for addition, adjust.
            for note in notes_to_add:
                if (
                    note.pitch == previous_note.pitch
                    and note.start == previous_note.start
                    and note.end == previous_note.end
                ):
                    note.end = min(note.end, start_time)
                    
            previous_note.end = min(previous_note.end, start_time)
            # if previous_note.end <= previous_note.start:
            #     print(
            #         f"duration3: {previous_note.end - previous_note.start}", flush=True
            #     )
            
    

    return duration, start_time, notes_to_remove, notes_to_add


# defines the pitch delta for the next note for overlap case.
def gather_pitches_and_adjust_pitch(inst, idx, stdev_pitch_delta, rng):
    # if this is the last note, just get a random pitch delta
    pitch_delta = int(rng.normal(loc=0, scale=stdev_pitch_delta))
    if idx >= len(inst.notes) - 1:
        while -1 < pitch_delta < 1:      
            pitch_delta = int(rng.normal(loc=0, scale=stdev_pitch_delta))
        return pitch_delta

    note = inst.notes[idx]
    # Initialize the list of pitches with the current note's pitch
    next_note_pitches = [note.pitch]
    next_note_index = idx + 1

    # Collect all pitches starting at the same time as the current note
    while (
        next_note_index < len(inst.notes) - 1
        and inst.notes[next_note_index].start == note.start
    ):
        print("while 1")
        next_note_pitches.append(inst.notes[next_note_index].pitch)
        next_note_index += 1
    # after the while loop, next_note_index is the index of the next note with a different start time
    if next_note_index < len(inst.notes) - 1:
        # Collect pitches of the next distinct start time note
        next_note = inst.notes[next_note_index]
        next_note_pitches.append(next_note.pitch)
        next_next_note_index = next_note_index + 1
        # Collect all pitches starting at the same time as the next note as well 
        while (
            next_next_note_index < len(inst.notes) - 1
            and inst.notes[next_next_note_index].start == next_note.start
        ):
            print("while 2")
            next_note_pitches.append(inst.notes[next_next_note_index].pitch)
            next_next_note_index += 1

    # Adjust the pitch using a normal distribution until a pitch + delta is found that is not in the list
    pitch_delta = 0
    while -1 < pitch_delta < 1 or note.pitch + int(pitch_delta) in next_note_pitches:
        print("while 3")
        pitch_delta = int(rng.normal(loc=0, scale=stdev_pitch_delta))

    return pitch_delta


def add_screwups(
    midi,
    lambda_occur=0.05,
    stdev_pitch_delta=1,
    mean_duration=1,
    stdev_duration=0.02,
    allow_overlap=False,
    fixed_screwup_type=None,
    config=get_config(),
):
    """
    Add screwups to the MIDI file by introducing mistakes in the notes.

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI object to modify.
        lambda_occur (float): The rate parameter for the Poisson distribution
            determining the rate of screwup occurrences per note.
        stdev_pitch_delta (float): The standard deviation of the pitch delta
            used to introduce pitch mistakes.
        mean_duration (float): The mean duration used to introduce extra notes
            or timing inaccuracies.
        stdev_duration (float): The standard deviation of the duration used to
            introduce extra notes or timing inaccuracies.
        allow_overlap (bool, optional): Whether to allow overlap between notes.
            Defaults to False.
        fixed_screwup_type (int, optional): The fixed screwup type to apply to
            all notes. If None, a random screwup type will be chosen for each note.
            Defaults to None.
        config (dict, optional): The configuration parameters for augmentation.
            Defaults to get_config().

    Returns:
        None
    """
    # Get tempo changes
    tempo_change_times, tempi = midi.get_tempo_changes()
    # print(tempo_change_times)

    # Create deep copies of the original MIDI for extra and removed notes
    midi_extra_notes = copy.deepcopy(midi)
    midi_removed_notes = copy.deepcopy(midi)
    midi_correct_notes = copy.deepcopy(midi)

    # Clear all notes in midi_extra_notes as it will be filled with new notes only
    for instrument in midi_extra_notes.instruments:
        instrument.notes.clear()
    for instrument in midi_removed_notes.instruments:
        instrument.notes.clear()
    for instrument in midi_correct_notes.instruments:
        instrument.notes.clear()
    print(midi.instruments)
    for inst in midi.instruments:
        inst.notes.sort(key=lambda x: x.start)
        for note in inst.notes:
            if note.end <= note.start:
                # we pre remove notes that have end time less than start time
                inst.notes.remove(note)

    for inst_index, inst in enumerate(midi.instruments):
        inst.notes.sort(key=lambda x: x.start)  # Ensure notes are sorted

        num_notes = len(inst.notes)
        if num_notes == 0:
            continue
        base_size = min(int(np.ceil(lambda_occur * num_notes)), num_notes)

        # Determine the half-range for variation
        half_range = base_size // 2

        # Calculate a random adjustment within +/- half_range
        # Ensure that size never goes below 0 or above num_notes
        # Size is for number of error indices.
        size_adjustment = rng.integers(-half_range, half_range + 1)
        size = max(0, min(num_notes, base_size + size_adjustment))
        error_indices = rng.choice(num_notes, size=size, replace=False)
        error_indices.sort()

        notes_to_add = []
        notes_to_remove = []

        for idx in error_indices:

            if idx >= len(inst.notes):
                print(f"idx: {idx} is greater than len(inst.notes): {len(inst.notes)}")

                break

            note = inst.notes[idx]
            if note.end <= note.start:
                inst.notes.remove(note)
                continue
            note_duration = note.end - note.start
            if fixed_screwup_type is not None:
                screwup_type = fixed_screwup_type
            else:
                screwup_type = rng.integers(
                    0, 16  # 4 to include timing inaccuracies
                )  # Now includes 0, 1, 2, 3, 4 as screwup types
                # Use bitwise ops to allow multiple types of events to happen at once
            if screwup_type != 0:
                notes_to_remove, notes_to_add = add_timing_inaccuracies(
                    rng,
                    inst,
                    note,
                    idx,
                    tempo_change_times,
                    tempi,
                    mean_duration,
                    stdev_duration,
                    stdev_pitch_delta,
                    allow_overlap,
                    config,
                    midi,
                    notes_to_remove,
                    notes_to_add,
                )

            if screwup_type == 0:  # Didn't play notes
                notes_to_remove.append(note)
                # inst.notes.remove(note)

            if screwup_type == 1:  # Plays wrong note
                note_removed = copy.deepcopy(note)
                notes_to_remove.append(note_removed)
                note_added = copy.deepcopy(note)

                if allow_overlap is False:
                    if idx < len(inst.notes) - 1:
                        next_note_pitch = inst.notes[idx + 1].pitch
                    else:
                        next_note_pitch = note.pitch
                    pitch_delta = 0
                    while (
                        -1 < pitch_delta < 1
                        or note.pitch + int(pitch_delta) == next_note_pitch
                    ):
                        pitch_delta = int(rng.normal(loc=0, scale=stdev_pitch_delta))
                else:
                    pitch_delta = gather_pitches_and_adjust_pitch(
                        inst, idx, stdev_pitch_delta, rng
                    )
                note_added.pitch = np.clip(note.pitch + int(pitch_delta), 0, 127)
                notes_to_add.append(note_added)

                # removed then appended later to avoid stacking errors
                # inst.notes.remove(note)

            elif screwup_type == 2:  # Messed up pitches and player fixes it quickly
                # Note is note missing in this case so we don't need to remove it. Maybe late?
                note_error = copy.deepcopy(note)

                if allow_overlap is False:
                    if idx < len(inst.notes) - 1:
                        next_note_pitch = inst.notes[idx + 1].pitch
                    else:
                        next_note_pitch = note.pitch
                    pitch_delta = 0
                    while (
                        -1 < pitch_delta < 1
                        or note.pitch + int(pitch_delta) == next_note_pitch
                    ):
                        pitch_delta = int(rng.normal(loc=0, scale=stdev_pitch_delta))
                else:
                    pitch_delta = gather_pitches_and_adjust_pitch(
                        inst, idx, stdev_pitch_delta, rng
                    )
                initial_note_dur = note_duration / 8.0 + rng.uniform(
                    low=-1 * note_duration / 32.0, high=(note_duration / 8.0) * 3.0
                )
                note_error.end = note_error.start + initial_note_dur
                note_error.pitch = np.clip(note.pitch + int(pitch_delta), 0, 127)
                notes_to_add.append(note_error)
                note.start = note.start + initial_note_dur

            if screwup_type == 3:  # Add extra notes
                notes_list, notes_to_remove, notes_to_add = get_extra_notes(
                    rng,
                    inst,
                    note,
                    idx,
                    mean_duration,
                    stdev_duration,
                    stdev_pitch_delta,
                    allow_overlap,
                    notes_to_remove,
                    notes_to_add,
                )
                for note2 in notes_list:
                    notes_to_add.append(note2)

            if screwup_type >= 4:
                # Added to every note, this is just a placeholder for pure timing error
                pass
        # TODO: tempo between midi org midi and mid of removed notes is not the same
        notes_to_remove_set = set(
            (note.pitch, note.start, note.end) for note in notes_to_remove
        )

        # Then filter the notes
        inst.notes = [
            note
            for note in inst.notes
            if (note.pitch, note.start, note.end) not in notes_to_remove_set
        ]
        # There might be some time discrepency here that are removing correct notes.
        # We want to maintain those correct notes as correct, 
        for note in notes_to_remove:

            midi_removed_notes.instruments[inst_index].notes.append(note)

        # print(
        #     f"After removal, instrument {inst_index} has {len(midi_removed_notes.instruments[inst_index].notes)} removed notes"
        # )

        # sort midi_removed_notes
        midi_removed_notes.instruments[inst_index].notes.sort(key=lambda x: x.start)

        # get correct notes
        inst.notes.sort(key=lambda x: x.start)
        # get a copy of the notes
        correct_notes = copy.deepcopy(inst.notes)
        for note in correct_notes:
            midi_correct_notes.instruments[inst_index].notes.append(note)

        # sort midi_correct_notes
        midi_correct_notes.instruments[inst_index].notes.sort(key=lambda x: x.start)

        # Add notes to midi_extra_notes
        # make notes_to_add mono if allow_overlap is False
        if allow_overlap is False:
            # sort notes_to_add
            notes_to_add.sort(key=lambda x: x.start)
            for i in range(len(notes_to_add) - 1):
                if notes_to_add[i].end > notes_to_add[i + 1].start:
                    notes_to_add[i].end = notes_to_add[i + 1].start

        for note in notes_to_add:
            inst.notes.append(note)
            midi_extra_notes.instruments[inst_index].notes.append(note)
        # sort midi_extra_notes
        # print(f" adding {len(notes_to_add)} notes")
        midi_extra_notes.instruments[inst_index].notes.sort(key=lambda x: x.start)
        # print(
        #     f"midi_extra_notes has {len(midi_extra_notes.instruments[inst_index].notes)} notes"
        # )

        # print(f"Removing {len(notes_to_remove)} notes")
        # print(notes_to_remove)
        # First, create a set of hashable representations of the notes to remove

        # Add notes to midi_extra_notes

        inst.notes.sort(key=lambda x: x.start)
        # comment this out for bug
        # if allow_overlap is False:
        #     print("Adjusting for overlap")
        #     make_instrument_mono(inst)
        # TODO need to make new midis are sorted and mono
    return midi, midi_extra_notes, midi_removed_notes, midi_correct_notes


def get_extra_notes(
    rng,
    inst,
    note,
    idx,
    mean_duration,
    stdev_duration,
    stdev_pitch_delta,
    allow_overlap,
    notes_to_remove,
    notes_to_add,
):
    notes = []
    note_duration = note.end - note.start
    variance_duration = stdev_duration**2

    # Parameters for the gamma distribution
    shape = mean_duration**2 / variance_duration
    scale = variance_duration / mean_duration

    # Generating a random value from the gamma distribution
    duration_var = rng.gamma(shape, scale)
    # print(f"duration_var: {duration_var}", flush=True)
    # randomize note duration by a gamma distribution
    # truncate the duration to be within 0.5 * note_duration and 2 * note_duration
    duration = max(1 / 2 * note_duration, note_duration * duration_var)
    duration = min(duration, note_duration * 2)
    
    next_note_idx = idx + 1
    next_note_start = (
        inst.notes[next_note_idx].start if next_note_idx < len(inst.notes) - 1 else note.end
    )
    if next_note_start < note.start:
        next_note_start = note.start  # Note: if timing error is supposed to be created, this would cancel it out.

    start_time = rng.uniform(low=note.start, high=next_note_start)

    # if idx < len(inst.notes) - 1:
    #     next_note_pitch = inst.notes[idx + 1].pitch
    # else:
    #     next_note_pitch = note.pitch
    if allow_overlap is False:
        if idx < len(inst.notes) - 1:
            next_note_pitch = inst.notes[idx + 1].pitch
        else:
            next_note_pitch = note.pitch
        pitch_delta = 0
        while -1 < pitch_delta < 1 or note.pitch + int(pitch_delta) == next_note_pitch:
            pitch_delta = int(rng.normal(loc=0, scale=stdev_pitch_delta))
    else:
        pitch_delta = gather_pitches_and_adjust_pitch(inst, idx, stdev_pitch_delta, rng)
    # potential rare bug if pitch is already at clip boundary
    new_pitch = np.clip(note.pitch + pitch_delta, 0, 127)

    # Debugging: Output generated values before adding the note
    # print(f"Original note duration: {note_duration}")
    # print(f"Randomized duration for extra note: {duration}")
    # print(f"Chosen start time for extra note: {start_time}")
    # print(f"Chosen pitch for extra note: {new_pitch}")
    if allow_overlap is False:
        print("Adjusting for overlap")
        # note.end = start_time
        duration, start_time, notes_to_remove, notes_to_add = adjust_for_overlap(
            inst, idx, duration, start_time, notes_to_remove, notes_to_add
        )

    # Debugging: Output final values before adding the note
    # print(f"Final duration for extra note: {duration}", flush=True)
    # print(
    #     f"Final start and end times for extra note: {start_time}, {start_time + duration}",
    #     flush=True,
    # )

    # Add the new note
    extra_note = pm.Note(
        velocity=note.velocity,
        pitch=new_pitch,
        start=start_time,
        end=start_time + duration,
    )
    notes.append(extra_note)

    # Debugging: Confirm addition and sorting
    # print(f"Added extra note: {extra_note}")
    # inst.notes.sort(key=lambda x: x.start)
    # if allow_overlap is False:
    #     make_instrument_mono(inst)
    # print(f"Notes sorted. Number of notes for instrument: {len(inst.notes)}")

    return notes, notes_to_remove, notes_to_add


def add_timing_inaccuracies(
    rng,
    inst,
    note,
    idx,
    tempo_change_times,
    tempi,
    mean_duration,
    stdev_duration,
    stdev_pitch_delta,
    allow_overlap,
    config,
    midi,
    notes_to_remove,
    notes_to_add,
):

    note_duration = note.end - note.start
    variance_duration = stdev_duration**2

    # Parameters for the gamma distribution
    shape = mean_duration**2 / variance_duration
    scale = variance_duration / mean_duration

    # Generating a random value from the gamma distribution such the duration is within 0.5 * note_duration and 2 * note_duration
    duration_var = rng.gamma(shape, scale)
    # print(f"duration_var: {duration_var}", flush=True)
    # randomize note duration by a gamma distribution
    # truncate the duration to be within 0.5 * note_duration and 2 * note_duration
    duration = max(1 / 2 * note_duration, note_duration * duration_var)
    duration = min(duration, note_duration * 2)

    # current_tempo = midi.estimate_tempo()

    # error_range = get_thirty_second_note_duration(
    #     current_tempo, midi
    # )  # for timing, start time should be +- 1/32 of a beat for a decent player

    # if note.start > 0.0:
    #     tempo_segment = np.flatnonzero(tempo_change_times < note.start)[-1]
    # else:
    #     tempo_segment = 0
    # current_tempo = tempi[tempo_segment]
    # error_range = get_thirty_second_note_duration(
    #     current_tempo, midi
    # )  # for timing, start time should be +- 1/32 of a beat for a decent player
    # # randomize note duration by a gamma distribution

    # Generate a timing shift using a normal distribution
    timing_shift_seconds = rng.normal(
        loc=config["expressive_timing_mean_ms"]
        / 1000.0,  # we use zero for equal chance of early and late
        scale=config["expressive_timing_std_ms"] / 1000.0,
    )
    # Lmit the timing shift to not reorder the notes. 
    if not allow_overlap:
        print(f"Adjusting for overlap for note {idx}")
        if idx < len(inst.notes) - 1 and idx > 0:
            next_note = inst.notes[idx + 1]
            previous_note = inst.notes[idx - 1]
            while (timing_shift_seconds >= (next_note.start - note.start)) or (
                timing_shift_seconds <= (previous_note.start - note.start)
            ):
                timing_shift_seconds = rng.normal(
                    loc=config["expressive_timing_mean_ms"]
                    / 1000.0,  # we use zero for equal chance of early and late
                    scale=config["expressive_timing_std_ms"] / 1000.0,
                )
        elif idx == 0:
            next_note = inst.notes[idx + 1]
            while timing_shift_seconds >= (next_note.start - note.start):
                timing_shift_seconds = rng.normal(
                    loc=config["expressive_timing_mean_ms"]
                    / 1000.0,  # we use zero for equal chance of early and late
                    scale=config["expressive_timing_std_ms"] / 1000.0,
                )
        elif idx == len(inst.notes) - 1:
            previous_note = inst.notes[idx - 1]
            while timing_shift_seconds <= (previous_note.start - note.start):
                timing_shift_seconds = rng.normal(
                    loc=config["expressive_timing_mean_ms"]
                    / 1000.0,  # we use zero for equal chance of early and late
                    scale=config["expressive_timing_std_ms"] / 1000.0,
                )
        else:
            timing_shift_seconds = 0
    else:
        # Overlap is allowed
        timing_shift_seconds = rng.normal(
            loc=config["expressive_timing_mean_ms"]
            / 1000.0,  # we use zero for equal chance of early and late
            scale=config["expressive_timing_std_ms"] / 1000.0,
        )
        
    note.start += timing_shift_seconds

    note.end = note.start + duration
    # Apply the timing shift without labelling it an error if it's within the bounds of a 32nd note
    # if abs(timing_shift_seconds) <= error_range:
    #     note.start += timing_shift_seconds

    #     note.end = note.start + duration
    # else:
    #     # Handle timing error outside of acceptable bounds
    #     # print(
    #     #     f"Timing error for note {note.pitch} at {note.start} with shift {timing_shift_seconds}s"
    #     # )
    #     note.start += timing_shift_seconds
    #     note.end = note.start + duration
    if allow_overlap is False:
        # Adjust for potential overlap with adjacent notes
        # Adjust duration of previous note to avoid overlap.
        print(f"Adjusting for overlap for note {idx}")
        if idx > 0:
            previous_note = inst.notes[idx - 1]

            if note.start < previous_note.end:

                # Adjust previous note to avoid overlap
                for n in notes_to_remove:
                    if (
                        n.pitch == previous_note.pitch
                        and n.start == previous_note.start
                        and n.end == previous_note.end
                    ):
                        n.end = min(previous_note.end, note.start)
                # check for overlap in notes to add and adjust accordingly
                for n in notes_to_add:

                    if (
                        n.pitch == previous_note.pitch
                        and n.start == previous_note.start
                        and n.end == previous_note.end
                    ):
                        n.end = min(previous_note.end, note.start)
                # key operation
                previous_note.end = min(previous_note.end, note.start)
                # if previous_note.end <= previous_note.start:
                #     print(
                #         f"duration1: {previous_note.end - previous_note.start}",
                #         flush=True,
                #     )
                    

        if idx < len(inst.notes) - 1:
            next_note = inst.notes[idx + 1]
            if note.end > next_note.start:
                
                # Adjust current note to avoid overlap
                note.end = min(note.end, next_note.start)

    return notes_to_remove, notes_to_add


def augment_mistakes(midi, overlap=True):
    pb = False  # TODO: pitch bends mistakes should be gotten from the actual complete midi file
    for inst in midi.instruments:
        if "piano" in inst.name.lower():
            pb = False
            overlap = True

    lambda_occur = rng.uniform(0.1, 0.4)
    # stdev_pitch_delta= rng.uniform(1, 3)
    midi, midi_extra_notes, midi_removed_notes, midi_correct_notes = add_screwups(
        midi=midi,
        lambda_occur=lambda_occur,
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
    return midi, midi_extra_notes, midi_removed_notes, midi_correct_notes
