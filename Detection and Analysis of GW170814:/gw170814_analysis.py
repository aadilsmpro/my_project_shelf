from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

'''
Fetching Data from GWPY Open Data Set For H1 and L1 Detectors
'''

print("Fetching 32s data snippet...")
event_gps = 1186741861
h1 = TimeSeries.fetch_open_data('H1', event_gps - 16, event_gps + 16)
l1 = TimeSeries.fetch_open_data('L1', event_gps - 16, event_gps + 16)

print("Success! Data loaded.")

'''
Plotting Data from GWOSC
'''

event_gps = 1186741861
h1 = TimeSeries.fetch_open_data('H1', event_gps - 16, event_gps + 16)
l1 = TimeSeries.fetch_open_data('L1', event_gps - 16, event_gps + 16)

print("Plotting data...")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(h1.times, h1.value, color='#ee0000', label='Hanford (H1) - Raw', alpha=0.8)
ax.plot(l1.times, l1.value, color='#4ba6ff', label='Livingston (L1) - Raw', alpha=0.8)

# x-axis limits set to focus around the event
ax.set_xlim(event_gps - 0.5, event_gps + 0.5)

ax.set_ylabel('Strain')
ax.set_xlabel('GPS Time (seconds)')
ax.legend(loc='upper left')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()

'''
Plotting Whitened Data
'''

h1_white = h1.whiten(fftlength=4, overlap=2)
l1_white = l1.whiten(fftlength=4, overlap=2)

print("Plotting whitened data...")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(h1_white.times, h1_white.value, color='#ee0000', label='Hanford (H1) - Whitened', alpha=0.8)
ax.plot(l1_white.times, l1_white.value, color='#4ba6ff', label='Livingston (L1) - Whitened', alpha=0.8)

# Set x-axis limits to focus around the event
ax.set_xlim(event_gps + 0.4, event_gps + 0.7)

ax.set_ylabel('Strain (Whitened)')
ax.set_xlabel('GPS Time (seconds)')
ax.legend(loc='upper left')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()

'''
Plotting Bandpassed Data
'''

h1_clean = h1_white.bandpass(20, 300)
l1_clean = l1_white.bandpass(20, 300)

print("Plotting bandpassed data")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Set x-axis limits to focus around the event
zoom = (event_gps + 0.4, event_gps + 0.7)

ax.plot(h1_clean.times, h1_clean.value, color='#ee0000', label='Hanford (H1)', alpha=0.8)
ax.plot(l1_clean.times, l1_clean.value, color='#4ba6ff', label='Livingston (L1)', alpha=0.8)

ax.set_xlim(*zoom)
ax.set_ylabel('Strain ')
ax.set_xlabel('GPS Time (seconds)')
ax.legend(loc='upper left')
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()

'''
Doing Q-transform and Plotting Q-transform Spectrogram
'''

hq = h1.q_transform(outseg=(event_gps + 0.4, event_gps + 0.7), qrange=(4, 8))

print("Generating Frequency Map (Spectrogram)...")

fig = hq.plot()
ax = fig.gca()
ax.set_yscale('log')
ax.set_ylim(20, 500)  # Standard frequency range for black holes
ax.set_ylabel('Frequency (Hz)')
ax.colorbar(label='Signal Energy')

plt.show()

'''
Finding Merger Time and Time When Chirp Signal Frequency Crossed 100 Hz
'''

def find_merger_from_data(hq):
    """
    Finds the absolute maximum energy point. 
    (Can sometimes be fooled by high-energy low-freq artifacts).
    """
    data_array = np.abs(hq.value) 
    # Checking orientation 
    if data_array.shape[0] == len(hq.frequencies):
        freq_idx, time_idx = np.unravel_index(np.argmax(data_array), data_array.shape)
    else:
        time_idx, freq_idx = np.unravel_index(np.argmax(data_array), data_array.shape)
    
    return hq.times[time_idx].value, hq.frequencies[freq_idx].value, data_array.max()

def find_physical_merger(hq, freq_threshold=150):
    """
    Finds the merger by looking for the peak energy specifically at the 
    higher frequency end of the chirp (above freq_threshold).
    Fixes the IndexError by detecting the correct data orientation.
    """
    data_array = np.abs(hq.value)
    freqs = hq.frequencies.value
    
    # Filtering for frequencies above threshold (150 Hz)
    high_freq_mask = freqs >= freq_threshold
    
    # Handle Transposition: 
    # If axis 0 size matches frequencies, use it. Otherwise, use axis 1.
    if data_array.shape[0] == len(freqs):
        high_freq_data = data_array[high_freq_mask, :]
        sub_freq_idx, time_idx = np.unravel_index(np.argmax(high_freq_data), high_freq_data.shape)
        actual_freq_idx = np.where(high_freq_mask)[0][sub_freq_idx]
    else:
        # Data is likely (times, frequencies)
        high_freq_data = data_array[:, high_freq_mask]
        time_idx, sub_freq_idx = np.unravel_index(np.argmax(high_freq_data), high_freq_data.shape)
        actual_freq_idx = np.where(high_freq_mask)[0][sub_freq_idx]
    
    merger_time = hq.times[time_idx].value
    merger_freq = hq.frequencies[actual_freq_idx].value
    peak_energy = np.max(high_freq_data)

    return merger_time, merger_freq, peak_energy

def find_time_at_frequency(hq, target_freq, proximity_time=None):
    """
    Finds the time when the signal energy is strongest for a specific frequency.
    
    If proximity_time is provided (e.g., the merger time), it looks for the 
    local peak closest to that time to avoid picking up noise at 0.43s.
    """
    freqs = hq.frequencies.value
    freq_idx = (np.abs(freqs - target_freq)).argmin()
    actual_freq = freqs[freq_idx]
    data_array = np.abs(hq.value)

    # Orientation check for 1D slice extraction
    if data_array.shape[0] == len(freqs):
        energy_at_freq = data_array[freq_idx, :]
    else:
        energy_at_freq = data_array[:, freq_idx]
    
    times = hq.times.value
    
    if proximity_time is not None:
        # Defining a window around the merger (within 0.15s before merger)
        # Chirps always move from lower to higher freq
        search_window_mask = (times > proximity_time - 0.15) & (times <= proximity_time)
        
        # If the window is valid, find the peak within that specific window
        if np.any(search_window_mask):
            window_indices = np.where(search_window_mask)[0]
            windowed_energy = energy_at_freq[search_window_mask]
            local_peak_idx = window_indices[np.argmax(windowed_energy)]
            time_idx = local_peak_idx
        else:
            time_idx = np.argmax(energy_at_freq)
    else:
        time_idx = np.argmax(energy_at_freq)
        
    return hq.times[time_idx].value, actual_freq, energy_at_freq[time_idx]

def print_legible_results(merger_time, merger_freq, energy, label="PEAK ANALYSIS"):
    """
    Formats the raw numerical output into a human-readable summary.
    """
    print("\n" + "="*50)
    print(f"{label}")
    print("="*50)
    print(f"{'GPS Time:':<25} {merger_time:.6f}")
    print(f"{'Frequency:':<25} {float(merger_freq):.2f} Hz")
    energy_val = getattr(energy, 'value', energy)
    print(f"{'Signal Energy:':<25} {float(energy_val):.2f}")
    print("="*50 + "\n")


# Getting the merger time
m_time, m_freq, m_energy = find_physical_merger(hq, freq_threshold=200)
print_legible_results(m_time, m_freq, m_energy, label="PHYSICAL MERGER (HIGH-FREQ PEAK)")

# Use m_time as proximity to find the correct 100Hz crossing
t_at_f, actual_f, f_energy = find_time_at_frequency(hq, 100, proximity_time=m_time)
print_legible_results(t_at_f, actual_f, f_energy, label=f"STRENGTH AT {actual_f:.1f} Hz")

'''
Calculating Chirp Mass(M)
'''

# Constants
c = 3e8          # Speed of light
G = 6.67e-11     # Gravity
M_sun = 1.989e30 # Mass of the sun

# Parameters selected
f_measure = actual_f   # Frequency in Hz 
tau = m_time - t_at_f  # Time remaining until the crash (seconds)

# Calculation
numerator = c**3
denominator = G
term1 = (5 / (256 * tau))**(3/5)
term2 = (np.pi)**(-8/5)
term3 = f_measure**(-8/5)

chirp_mass_kg = (numerator / denominator) * term1 * term2 * term3
chirp_mass_solar = chirp_mass_kg / M_sun

print("="*40)
print(f"ESTIMATED CHIRP MASS: {chirp_mass_solar:.2f} Solar Masses")
print("="*40)
