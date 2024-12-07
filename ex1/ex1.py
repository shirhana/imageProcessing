import matplotlib.pyplot as plt
import mediapy as media
import numpy as np

def calculate_histogram(frame):
    """
    Calculate the histogram of a binary image.
    :param frame: black and white binary image as a numpy array (values 0 or 255)
    :return: Histogram of pixel intensities (array of size 256)
    """
    # Flatten the frame to a 1D array and calculate the histogram
    histogram, _ = np.histogram(frame, bins=256, range=(0, 256))
    return histogram

def compare_histograms(hist1, hist2):
    """
    Compare two histograms and return a difference score.
    :param hist1: First histogram (normalized)
    :param hist2: Second histogram (normalized)
    :return: Difference score (higher values represent greater difference)
    """
    # Normalize histograms by dividing by their sum (L1 normalization)
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # compute relative differences if you want to reduce the effect of smaller changes
    relative_diff = np.sum(np.abs(hist1 - hist2) / (hist1 + hist2 + 1e-10))  # Relative difference
    
    return relative_diff

def plot_histogram_comparison(hist1, hist2, frame_idx_1, frame_idx_2):
    """
    Plot the histograms of two frames to compare them visually.
    :param hist1: Histogram of the first frame
    :param hist2: Histogram of the second frame
    :param frame_idx_1: Index of the first frame
    :param frame_idx_2: Index of the second frame
    """
    plt.figure(figsize=(10, 6))
    plt.plot(hist1, label=f'Frame {frame_idx_1} Histogram', color='b')
    plt.plot(hist2, label=f'Frame {frame_idx_2} Histogram', color='r')
    plt.title(f"Histogram Comparison between Frame {frame_idx_1} and Frame {frame_idx_2}")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_histogram_difference(hist_differences, scene_cut_frames):
    # Plot the histogram differences
    plt.figure(figsize=(10, 5))
    plt.plot(hist_differences, label="Histogram Differences", color="blue")
    plt.xlabel("Frame Number")
    plt.ylabel("Histogram Difference")
    plt.title("Histogram Differences Across Frames")
    plt.axvline(scene_cut_frames[0], color='red', linestyle='--', label="Scene Cut Frame")
    plt.legend()
    plt.show()

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame numbers for which the scene cut was detected 
             (i.e., the frames with the largest change in histogram).
    """
    
    # Read video as a numpy array
    video_frames = media.read_video(video_path)
    
    prev_frame_hist = None
    max_diff = -np.inf
    scene_cut_frames = (None, None)
    hist_differences = []

    
    for i, frame in enumerate(video_frames):
        # Convert frame to grayscale by averaging the color channels
        gray_frame = np.mean(frame, axis=-1).astype(np.uint8)

        if str(video_type) == '2':
            # Convert to binary (black and white) by applying a threshold (e.g., 128)
            binary_frame = (gray_frame > 128).astype(np.uint8) * 255  # Apply threshold to create a binary image
        else:
            binary_frame = gray_frame

        # Calculate histogram for the current binary frame
        current_hist = calculate_histogram(binary_frame)

        # Calculate difference in histograms between consecutive frames
        if prev_frame_hist is not None:
            # Compare histograms (Euclidean distance in this case)
            hist_diff = compare_histograms(prev_frame_hist, current_hist)
            hist_differences.append(hist_diff)
            
            # If the histogram difference exceeds the current maximum, store these frames
            if hist_diff > max_diff:
                max_diff = hist_diff
                scene_cut_frames = (i - 1, i)

                # plot_histogram_comparison(prev_frame_hist, current_hist, i - 1, i)

        # Store the current histogram for the next iteration
        prev_frame_hist = current_hist

    # plot_histogram_difference(hist_differences, scene_cut_frames)

    # Display the frames before and after the scene cut
    if scene_cut_frames[0] is not None and scene_cut_frames[1] is not None:
        print(f"Scene cut detected between frames {scene_cut_frames[0]} and {scene_cut_frames[1]}")

    return scene_cut_frames
