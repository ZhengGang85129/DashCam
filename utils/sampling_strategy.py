import random
import math
import numpy as np
import cv2

def uniform_random_sampling(total_frames, num_frames, time_of_event=None, time_of_alert=None, fps=30):
    """
    Baseline random sampling function that uniformly samples frames from a video.

    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        time_of_event (float, optional): Timestamp of the accident event in seconds
        time_of_alert (float, optional): Timestamp of the alert in seconds
        fps (int, optional): Frames per second of the video, default is 30

    Returns:
        list: List of frame indices to extract
    """
    # Simple uniform interval calculation
    interval = max(1, math.floor(total_frames / num_frames))

    # Generate frame indices at regular intervals
    indices = list(range(0, total_frames, interval))[:num_frames]

    # If we don't have enough frames, duplicate the last ones
    while len(indices) < num_frames:
        indices.append(min(total_frames - 1, indices[-1] + 1))

    return indices


def alert_focused_sampling(total_frames, num_frames, time_of_event=None, time_of_alert=None, fps=30):
    """
    Advanced sampling function that focuses on the alert time, the progression to the event,
    and includes key frames based on the competition evaluation criteria.

    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        time_of_event (float): Timestamp of the accident event in seconds
        time_of_alert (float): Timestamp of the alert in seconds
        fps (int, optional): Frames per second of the video, default is 30

    Returns:
        list: List of frame indices to extract
    """
    # If no event/alert timestamps (non-accident video) or invalid timestamps, fall back to uniform sampling
    if time_of_event is None or time_of_alert is None or time_of_event <= 0 or time_of_alert <= 0:
        return uniform_random_sampling(total_frames, num_frames)

    # Convert timestamps to frame indices
    event_frame = min(total_frames - 1, int(time_of_event * fps))
    alert_frame = min(total_frames - 1, int(time_of_alert * fps))

    # Ensure alert_frame is before event_frame
    if alert_frame >= event_frame:
        alert_frame = max(0, event_frame - int(0.5 * fps))  # 0.5 seconds before event if invalid

    # Calculate frames at critical evaluation points (500ms, 1000ms, 1500ms before event)
    critical_frames = [
        max(0, event_frame - int(0.5 * fps)),  # 500ms before event
        max(0, event_frame - int(1.0 * fps)),  # 1000ms before event
        max(0, event_frame - int(1.5 * fps)),  # 1500ms before event
    ]

    # Distribution of frames
    # - 6 frames around alert time (the earliest prediction point)
    # - 4 frames between alert and event (progression)
    # - 2 frames at critical evaluation points
    # - 2 frames at event time
    # - 2 frames from earlier in the video (context)

    # Initialize with important frames
    sampled_indices = []

    # Add frames around alert time (6 frames)
    alert_window = 3  # frames before and after alert_frame
    for offset in range(-alert_window, alert_window):
        frame_idx = max(0, min(total_frames - 1, alert_frame + offset))
        if frame_idx not in sampled_indices:
            sampled_indices.append(frame_idx)

    # Add frames between alert and event (4 frames)
    if event_frame > alert_frame:
        interval = max(1, (event_frame - alert_frame) // 5)  # 5 intervals = 4 points between
        for i in range(1, 5):  # 4 points
            frame_idx = min(total_frames - 1, alert_frame + i * interval)
            if frame_idx not in sampled_indices:
                sampled_indices.append(frame_idx)

    # Add critical evaluation frames (500ms, 1000ms, 1500ms before event)
    for frame in critical_frames:
        if frame not in sampled_indices:
            sampled_indices.append(frame)

    # Add event frame and one after
    if event_frame not in sampled_indices:
        sampled_indices.append(event_frame)
    after_event = min(total_frames - 1, event_frame + 1)
    if after_event not in sampled_indices:
        sampled_indices.append(after_event)

    # Add some context frames from earlier in the video
    pre_alert_region = list(range(0, max(0, alert_frame - alert_window)))
    if pre_alert_region:
        # Add 2 frames from earlier parts
        early_frames = sorted(random.sample(pre_alert_region, min(2, len(pre_alert_region))))
        for frame in early_frames:
            if frame not in sampled_indices:
                sampled_indices.append(frame)

    # Sort the indices
    sampled_indices = sorted(sampled_indices)

    # If we have more frames than needed, prioritize keeping alert, critical points, and event frames
    if len(sampled_indices) > num_frames:
        # Keep alert frame, critical evaluation frames, and event frame
        must_keep = [alert_frame] + critical_frames + [event_frame]
        must_keep = [idx for idx in must_keep if 0 <= idx < total_frames]

        # Remove frames until we reach num_frames, but keep the must_keep frames
        removable_indices = [idx for idx in sampled_indices if idx not in must_keep]
        to_remove = len(sampled_indices) - num_frames

        # If we need to remove more frames than we have removable frames, we'll need to compromise
        if to_remove > len(removable_indices):
            # Sort the must_keep frames by importance (keeping alert and event)
            secondary_priority = [f for f in must_keep if f not in [alert_frame, event_frame]]
            to_remove_from_secondary = to_remove - len(removable_indices)

            # Remove the least important frames
            for _ in range(min(to_remove_from_secondary, len(secondary_priority))):
                least_important = secondary_priority.pop(0)
                must_keep.remove(least_important)
                removable_indices.append(least_important)

            # Recalculate how many to remove from removable indices
            to_remove = len(sampled_indices) - num_frames

        # Randomly select frames to remove
        to_remove_indices = random.sample(removable_indices, to_remove)
        sampled_indices = [idx for idx in sampled_indices if idx not in to_remove_indices]

    # If we still need more frames, add them from areas not yet covered
    while len(sampled_indices) < num_frames:
        # Find the largest gaps between consecutive frames
        gaps = [(sampled_indices[i+1] - sampled_indices[i], i)
               for i in range(len(sampled_indices)-1)]
        gaps.sort(reverse=True)  # Sort by gap size (largest first)

        if not gaps:  # No gaps found
            # Just duplicate the last frame
            sampled_indices.append(min(total_frames - 1, sampled_indices[-1] + 1))
        else:
            # Add a frame in the middle of the largest gap
            gap_size, gap_start_idx = gaps[0]
            frame_idx = sampled_indices[gap_start_idx] + gap_size // 2
            sampled_indices.append(frame_idx)
            sampled_indices.sort()  # Re-sort indices

    return sampled_indices

def test_sampling_functions():
    """
    Test function to demonstrate how the sampling functions work
    """
    # Test parameters
    total_frames = 300  # 10 seconds at 30fps
    num_frames = 16
    fps = 30

    # Test case 1: Non-accident video (no event/alert times)
    print("Test 1: Non-accident video (uniform sampling)")
    indices = uniform_random_sampling(total_frames, num_frames)
    print(f"Uniform sampling indices: {indices}")

    # Test case 2: Accident video with event at 8s and alert at 6s
    time_of_event = 8.0  # seconds
    time_of_alert = 6.0  # seconds
    print(f"\nTest 2: Accident video (event at {time_of_event}s, alert at {time_of_alert}s)")
    indices = alert_focused_sampling(total_frames, num_frames, time_of_event, time_of_alert, fps)

    # Convert indices back to timestamps for easier interpretation
    timestamps = [round(idx / fps, 2) for idx in indices]
    print(f"Alert-focused sampling indices: {indices}")
    print(f"Corresponding timestamps (seconds): {timestamps}")

    # Highlight which frames correspond to important events
    print(f"Alert frame included: {int(time_of_alert * fps) in indices}")
    print(f"Event frame included: {int(time_of_event * fps) in indices}")
    print(f"500ms before event included: {int((time_of_event - 0.5) * fps) in indices}")
    print(f"1000ms before event included: {int((time_of_event - 1.0) * fps) in indices}")
    print(f"1500ms before event included: {int((time_of_event - 1.5) * fps) in indices}")

if __name__ == "__main__":
    test_sampling_functions()
