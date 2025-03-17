import random
import math
import numpy as np

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


def alert_focused_sampling(total_frames, num_frames, time_of_event=None, time_of_alert=None, fps=30, video_duration_sec=10):
    """
    Advanced sampling function that focuses on the alert time and the progression towards
    the potential accident, adapted specifically for the competition requirements:
    1. EXCLUDES frames at or after the accident event
    2. Matches the expected test duration (approximately 10 seconds)
    3. Focuses on frames that would help predict accidents before they occur

    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        time_of_event (float): Timestamp of the accident event in seconds
        time_of_alert (float): Timestamp of the alert in seconds
        fps (int, optional): Frames per second of the video, default is 30
        video_duration_sec (int, optional): Target duration of sampled video in seconds, default is 10

    Returns:
        list: List of frame indices to extract
    """
    # If no event/alert timestamps (non-accident video) or invalid timestamps, fall back to uniform sampling
    # But constrain to match expected test video duration
    if time_of_event is None or time_of_alert is None or time_of_event <= 0 or time_of_alert <= 0:
        # For non-accident videos, we'll still sample a consistent duration
        max_frame = min(total_frames - 1, int(video_duration_sec * fps))
        return uniform_random_sampling(max_frame + 1, num_frames)

    # Convert timestamps to frame indices
    event_frame = min(total_frames - 1, int(time_of_event * fps))
    alert_frame = min(total_frames - 1, int(time_of_alert * fps))

    # Ensure alert_frame is before event_frame
    if alert_frame >= event_frame:
        alert_frame = max(0, event_frame - int(0.5 * fps))  # 0.5 seconds before event if invalid

    # IMPORTANT: We'll only consider frames BEFORE the event
    # This is crucial since the test videos are trimmed before accidents occur
    valid_end_frame = event_frame - 1  # Exclude the event frame itself

    # Set the start frame to maintain approximately 10-second duration
    # (matching expected test video duration)
    start_frame = max(0, valid_end_frame - int(video_duration_sec * fps) + 1)

    # Calculate frames at critical evaluation points (500ms, 1000ms, 1500ms before event)
    # These are key points used in the competition evaluation
    critical_frames = [
        max(start_frame, event_frame - int(0.5 * fps)),  # 500ms before event
        max(start_frame, event_frame - int(1.0 * fps)),  # 1000ms before event
        max(start_frame, event_frame - int(1.5 * fps)),  # 1500ms before event
    ]

    # Initialize with important frames
    sampled_indices = []

    # Add frames around alert time (6 frames), but only if they fall within our valid range
    alert_window = 3  # frames before and after alert_frame
    for offset in range(-alert_window, alert_window):
        frame_idx = max(start_frame, min(valid_end_frame, alert_frame + offset))
        if frame_idx not in sampled_indices:
            sampled_indices.append(frame_idx)

    # Add frames between alert and event (4 frames), ensuring we don't include the event itself
    if valid_end_frame > alert_frame:
        interval = max(1, (valid_end_frame - alert_frame) // 5)  # 5 intervals = 4 points between
        for i in range(1, 5):  # 4 points
            frame_idx = min(valid_end_frame, alert_frame + i * interval)
            if frame_idx not in sampled_indices:
                sampled_indices.append(frame_idx)

    # Add critical evaluation frames (500ms, 1000ms, 1500ms before event)
    for frame in critical_frames:
        if frame not in sampled_indices and frame <= valid_end_frame:
            sampled_indices.append(frame)

    # Add context frames from earlier in the video, within our valid range
    pre_alert_region = list(range(start_frame, max(start_frame, alert_frame - alert_window)))
    if pre_alert_region:
        # Add more context frames to better match the test scenario
        num_context = min(4, len(pre_alert_region))
        if num_context > 0:
            # Distribute context frames evenly across the pre-alert region
            step = max(1, len(pre_alert_region) // num_context)
            context_frames = [pre_alert_region[i] for i in range(0, len(pre_alert_region), step)][:num_context]
            for frame in context_frames:
                if frame not in sampled_indices:
                    sampled_indices.append(frame)

    # Sort the indices
    sampled_indices = sorted(sampled_indices)

    # If we have more frames than needed, prioritize keeping alert and critical points
    if len(sampled_indices) > num_frames:
        # Keep alert frame and critical evaluation frames
        must_keep = [alert_frame] + critical_frames
        must_keep = [idx for idx in must_keep if start_frame <= idx <= valid_end_frame]

        # Remove frames until we reach num_frames, but keep the must_keep frames
        removable_indices = [idx for idx in sampled_indices if idx not in must_keep]
        to_remove = len(sampled_indices) - num_frames

        # If we need to remove more frames than we have removable frames, we'll need to compromise
        if to_remove > len(removable_indices):
            # Sort the must_keep frames by importance (keeping alert as highest priority)
            secondary_priority = [f for f in must_keep if f != alert_frame]
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

        if not gaps:  # No gaps found or only one frame so far
            # Add frames from the valid range that aren't already included
            valid_range = list(range(start_frame, valid_end_frame + 1))
            remaining = [f for f in valid_range if f not in sampled_indices]

            if remaining:
                # Try to distribute remaining frames evenly
                step = max(1, len(remaining) // (num_frames - len(sampled_indices)))
                additional_frames = [remaining[i] for i in range(0, len(remaining), step)]

                for frame in additional_frames:
                    if frame not in sampled_indices:
                        sampled_indices.append(frame)
                        if len(sampled_indices) >= num_frames:
                            break
            else:
                # If we've used all valid frames, duplicate nearest to valid_end_frame
                # This is unlikely but handles edge cases
                nearest = min(sampled_indices, key=lambda x: abs(x - valid_end_frame))
                new_idx = max(start_frame, min(valid_end_frame, nearest + (-1 if nearest > valid_end_frame/2 else 1)))
                sampled_indices.append(new_idx)
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
    print(f"Event frame included: {int(time_of_event * fps) in indices}")  # Should be False with new implementation
    print(f"500ms before event included: {int((time_of_event - 0.5) * fps) in indices}")
    print(f"1000ms before event included: {int((time_of_event - 1.0) * fps) in indices}")
    print(f"1500ms before event included: {int((time_of_event - 1.5) * fps) in indices}")

    # Test case 3: Longer accident video with limited sampling window
    total_frames = 600  # 20 seconds at 30fps
    time_of_event = 18.0  # seconds - accident happens late in the video
    time_of_alert = 15.0  # seconds
    video_duration_sec = 10  # target 10-second samples
    print(f"\nTest 3: Longer accident video (event at {time_of_event}s, alert at {time_of_alert}s, 10s window)")
    indices = alert_focused_sampling(total_frames, num_frames, time_of_event, time_of_alert, fps, video_duration_sec)

    # Convert indices back to timestamps for easier interpretation
    timestamps = [round(idx / fps, 2) for idx in indices]
    print(f"Alert-focused sampling indices: {indices}")
    print(f"Corresponding timestamps (seconds): {timestamps}")

    # Verify we're excluding event frames and maintaining the right window
    event_frame = int(time_of_event * fps)
    window_start = max(0, event_frame - int(video_duration_sec * fps))
    print(f"Event frame ({event_frame}) included: {event_frame in indices}")  # Should be False
    print(f"Frames outside the 10s window included: {any(idx < window_start for idx in indices)}")  # Should be False
    print(f"Time span of sampled frames: {timestamps[-1] - timestamps[0]:.2f}s")  # Should be ≤ 10s


if __name__ == "__main__":
    test_sampling_functions()
