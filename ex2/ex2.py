import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from scipy.fft import fft
from pydub import AudioSegment
import math
import os

RESULTS = "results"

# Load audio files
def load_audio(file_path):
    rate, data = read(file_path)
    return rate, data

# Function to load the audio file
def load_audio_with_extra_details(file_path):
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples())
    # TOD - check the frame rate?
    return audio.frame_rate, samples, audio.sample_width, audio.channels

def add_watermark(samples, frame_rate, watermark_type="good"):
    # Add a low-amplitude as a defualt watermark
    amplitude = 1
        
    if watermark_type == "bad":
        # Add a high-amplitude as a watermark
        amplitude = 400
    
    duration = len(samples) / frame_rate
    t = np.linspace(0, duration, len(samples))
    watermark = amplitude * np.sin(2 * np.pi * 10000 * t)  # 10 kHz sine wave, low amplitude
    watermarked_audio = samples + watermark
    return np.clip(watermarked_audio, -32768, 32767).astype(np.int16)

# Function to save the audio file
def save_audio(file_path, frame_rate, watermarked_audio, sample_width, channels):
    output_audio = AudioSegment(
        watermarked_audio.tobytes(), 
        frame_rate=frame_rate, 
        sample_width=sample_width, 
        channels=channels
    )
    output_audio.export(file_path, format="wav")

def plot_frequency_spectrum(samples, frame_rate, title="Frequency Spectrum", save_path=None, freq_range=None):
    N = len(samples)
    fft_result = fft(samples)
    freqs = np.fft.fftfreq(N, 1 / frame_rate)
    magnitudes = np.abs(fft_result)
    
    # Plot the spectrum
    plt.figure(figsize=(10, 5))
    
    if freq_range:
        # Limit the frequency range for better visualization
        plt.plot(freqs[:N//2], magnitudes[:N//2])
        plt.xlim(freq_range)  # Limit x-axis to specified frequency range
    else:
        plt.plot(freqs[:N//2], magnitudes[:N//2])  # Only plot the positive frequencies
    
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.close()

# Load audio files
def load_audio_files(file_paths):
    audio_data = []
    sampling_rates = []
    for path in file_paths:
        rate, data = load_audio(file_path=path)
        sampling_rates.append(rate)
        audio_data.append(data)
    return sampling_rates, audio_data

# Compute frequency spectrum
def compute_frequency_spectrum(audio, sampling_rate):
    freq = np.fft.rfftfreq(len(audio), d=1/sampling_rate)
    spectrum = np.abs(np.fft.rfft(audio))
    return freq, spectrum

# Pad or truncate features
def pad_or_truncate(features, target_length):
    padded_features = []
    for feature in features:
        if len(feature) < target_length:
            feature = np.pad(feature, (0, target_length - len(feature)))
        else:
            feature = feature[:target_length]
        padded_features.append(feature)
    return np.array(padded_features)


# Function to compute and plot the spectrogram
def plot_spectrogram(audio_data, sample_rate, spectrogram_id, path):
    f, t, Sxx = spectrogram(audio_data, sample_rate)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram {spectrogram_id}')
    plt.colorbar(label='Power [dB]')
    plt.ylim(0, 23000)  # Limiting to high frequencies (for clarity)
    plt.savefig(path)
    return f, t, Sxx

# Function to detect peaks at a given frequency
def detect_peaks(f, t, Sxx, target_freq, threshold_dB, threshold_cycle_time):
    # Find the index of the closest frequency to the target frequency
    target_freq_idx = (np.abs(f - target_freq)).argmin()
    # Extract the magnitude at the target frequency over time
    magnitude_at_target_freq = 10 * np.log10(Sxx[target_freq_idx, :])
    times_above_threshold = t[magnitude_at_target_freq > threshold_dB]
    
    # add threshold for too nearest peaks 
    reduced = [times_above_threshold[0]]
    for value in times_above_threshold[1:]:
        # Check if the difference from the last added value exceeds the threshold
        if all(abs(value - kept_value) > threshold_cycle_time for kept_value in reduced):
            reduced.append(value)
    return np.array(reduced)

def calculate_times_above_threshold(f, t, Sxx, target_freq, threshold_dB, threshold_cycle_time):
    # Detect times where the magnitude exceeds the threshold
    times_above_threshold = detect_peaks(f, t, Sxx, target_freq, threshold_dB, threshold_cycle_time)    
    # Calculate time differences between consecutive peaks
    time_differences = np.diff(times_above_threshold)
    # Compute the average time difference (cycle time)
    cycle_time = np.mean(time_differences)
    num_of_peaks = len(times_above_threshold)

    return cycle_time, num_of_peaks

# Identify spectral peaks
def find_peaks_l(spectrum, threshold=0.1):
    max_val = max(spectrum)
    peaks = np.where(spectrum > max_val * threshold)[0]
    return peaks

def compute_frequency_spectrum(audio, sample_rate):
    # Perform FFT to compute the frequency spectrum of the audio
    N = len(audio)
    fft_result = np.fft.fft(audio)
    freqs = np.fft.fftfreq(N, 1 / sample_rate)
    magnitudes = np.abs(fft_result)
    return freqs, magnitudes

# Compare audio files
def analyze_speedup(file1, file2):
    rate1, audio1 = load_audio(file1)
    rate2, audio2 = load_audio(file2)

    # Frequency analysis and plot for File1 & File2
    plot_frequency_spectrum(audio1, rate1, title="Frequency Spectrum - File 1", save_path=os.path.join(RESULTS, "task3_file1_frequency_spectrum.png"))
    plot_frequency_spectrum(audio2, rate2, title="Frequency Spectrum - File 2", save_path=os.path.join(RESULTS, "task3_file2_frequency_spectrum.png"))

    # Step 1: Frequency analysis
    freq1, spectrum1 = compute_frequency_spectrum(audio1, rate1)
    freq2, spectrum2 = compute_frequency_spectrum(audio2, rate2)

    # Find spectral peaks
    peaks1 = freq1[find_peaks_l(spectrum1)]
    peaks2 = freq2[find_peaks_l(spectrum2)]

    # Step 2: Calculate speedup ratio (x) based on frequency scaling
    x_freq = peaks2[0] / peaks1[0]  # Assume peaks are ordered and comparable

    # Step 3: Time domain comparison
    len_ratio = len(audio1) / len(audio2)  # Compare signal lengths
    x_time = 1 / len_ratio  # Speed-up factor in time domain

    # Determine methods
    if np.isclose(x_freq, x_time, atol=0.1):  # Close match indicates frequency domain
        return f"File1: Time Domain, File2: Frequency Domain, Speed-up Factor (x): {x_time:.2f}"
    else:
        return f"File1: Frequency Domain, File2: Time Domain, Speed-up Factor (x): {x_freq:.2f}"


def execute_first_task(base_folder="Task 1"):
    print(f"***************************************************")
    print(f"TASK 1 RESULTS:")
    input_file = os.path.join(base_folder, "task1.wav")
    good_output_file = os.path.join(base_folder, "good_watermarked.wav")
    bad_output_file = os.path.join(base_folder, "bad_watermarked.wav")

    # Load audio
    frame_rate, samples, sample_width, channels = load_audio_with_extra_details(input_file)
    print(f"Loaded audio with frame rate: {frame_rate}, sample width: {sample_width}, channels: {channels}")

    # Plot the frequency spectrum of the original audio
    plot_frequency_spectrum(
        samples, frame_rate, title="Original Audio Frequency Spectrum", 
        save_path=os.path.join(RESULTS, "task1_original_audio_frequency_spectrum")
    )

    # Add watermarks
    good_watermarked = add_watermark(samples, frame_rate, watermark_type="good")
    bad_watermarked = add_watermark(samples, frame_rate, watermark_type="bad")
   
    # Plot the frequency spectrum of the "good" watermarked audio
    plot_frequency_spectrum(
        good_watermarked, frame_rate, title="Good Watermarked Audio Frequency Spectrum", 
        save_path=os.path.join(RESULTS, "good_wathermarked_audio_frequency_spectrum"),
        freq_range=(0, 2000)  # Focus on low frequencies (0-2 kHz)
    )

    # Plot the frequency spectrum of the "bad" watermarked audio
    plot_frequency_spectrum(
        bad_watermarked, frame_rate, title="Bad Watermarked Audio Frequency Spectrum", 
        save_path=os.path.join(RESULTS, "bad_wathermarked_audio_frequency_spectrum"),
        freq_range=(0, 22000) # Zoom out to 0-22 kHz to capture the high frequency
    )

    # Save watermarked audio
    save_audio(good_output_file, frame_rate, good_watermarked, sample_width, channels)
    save_audio(bad_output_file, frame_rate, bad_watermarked, sample_width, channels)

    print("Watermarked audio files have been created!")
    print(f"***************************************************\n\n\n")

def execute_second_task(base_folder="Task 2"):
    print(f"***************************************************")
    print(f"TASK 2 RESULTS:")

    base_threshold_cycle_time = 1.5
    threshold_cycle_time = base_threshold_cycle_time
    prev_cycle_time = None
    class_id = 0

    for i in range(9):
        filename = os.path.join(base_folder,f"{i}_watermarked.wav")  # Replace with your audio file path
        sample_rate, audio_data = load_audio(filename)
        # Plot the full spectrogram
        f, t, Sxx = plot_spectrogram(audio_data, sample_rate, spectrogram_id=f"{i}_watermarked", path=os.path.join(RESULTS,f"task2_{i}_spectogram.png"))
        # Set the frequency and threshold in dB ---> According to the figures which have seen.
        target_freq = 20000  # Target frequency (Hz)
        threshold_dB = -6
        
        # calculate with previous cycle time as threshold
        cycle_time, num_of_peaks = calculate_times_above_threshold(f, t, Sxx, target_freq, threshold_dB, threshold_cycle_time)

        # if the prev audio is in another category...
        if prev_cycle_time and math.trunc(prev_cycle_time*10)/10 != math.trunc(cycle_time*10)/10:
            class_id += 1
            # recalculate with base thereshold cycle time
            cycle_time, num_of_peaks = calculate_times_above_threshold(f, t, Sxx, target_freq, threshold_dB, base_threshold_cycle_time)

        # update threshold & pre cycle time
        threshold_cycle_time = cycle_time - 0.1
        prev_cycle_time = cycle_time
        print(f"file {os.path.basename(filename)} in class {class_id} with function: sin(2π * {num_of_peaks}/30 * x)")

def execute_third_task(base_folder="Task 3"):
    print(f"***************************************************")
    print(f"TASK 3 RESULTS:")
    # Paths to the audio files 
    file1 = os.path.join(base_folder, "task3_watermarked_method1.wav")
    file2 = os.path.join(base_folder, "task3_watermarked_method2.wav")

    result = analyze_speedup(file1, file2)
    print(result)
    print(f"***************************************************")

# Main workflow
if __name__ == "__main__":
    # TASK 1    
    execute_first_task()

    # TASK 2
    execute_second_task()

    # TASK 3
    execute_third_task()
