import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "data1.txt" #Time domain data

data = pd.read_csv(file_path,delim_whitespace=True, comment="%", names=["time", "Ref", "Sam"])

'''
Fourier Transform from Time to Frequency domain
'''

time = data["time"].values
ref_signal = data["Ref"].values
sam_signal = data["Sam"].values

dt = np.mean(np.diff(time))
fs = 1 / dt

freqs = np.fft.fftfreq(len(time), d=dt)
fft_ref = np.fft.fft(ref_signal)
fft_sam = np.fft.fft(sam_signal)

freql = len(freq)//2

# Plot for FFT'ed signal
plt.figure(figsize=(10, 5))
plt.plot(freqs[:freql], np.abs(fft_ref[:freql]), label="Ref (Magnitude)", color='b')
plt.plot(freqs[:freql], np.abs(fft_sam[:freql]), label="Sam (Magnitude)", color='r')
plt.xlabel("Frequency (THz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Reference and Sample Signal")
plt.legend()
plt.grid()
plt.show()

# Plot for raw signal
plt.figure(figsize=(10, 5))
plt.plot(time, ref_signal, label="Ref (Magnitude)", color='b')
plt.plot(time, sam_signal, label="Sam (Magnitude)", color='r')
plt.xlabel("Time (ps)")
plt.ylabel("Magnitude")
plt.title("Reference and Sample Signal")
plt.legend()
plt.grid()
plt.show()

print("Sampling Frequency (fs):", fs, "Hz")

'''
Calculation T(Ω) and Truncation of data
'''

Tomega = fft_sam / fft_ref

print(f"Original data points: {len(freqs)}")
valid_range_mask = (freqs >= 0.98) & (freqs <= 15.71)
freqs = freqs[valid_range_mask]
Tomega = Tomega[valid_range_mask]
print(f"Truncated data points (0-20 THz): {len(freqs)}")

freqs_hz = freqs * 1e12

# Plot of Tomega magnitude and phase
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freqs, np.abs(Tomega), color='green', linewidth=1.5)
plt.xlabel("Frequency (THz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs, np.angle(Tomega), color='green', linewidth=1.5)
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (radians)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

'''
Calculating the Complex Refractive Index of the sample
'''

# Physical constants and parameters
n_air = 1
d = 0.47e-3  # m
c = 299792458  # m/s
N = 3

def preprocess_phase_data(phase_data):
    """Remove 2π discontinuities from phase data"""
    return np.unwrap(phase_data)

def calculate_frequency_weights(freqs, snr_data=None, target_range=(1.587, 1.817)):
    if snr_data is not None:
        weights = snr_data / np.max(snr_data)
    else:
        weights = np.ones_like(freqs)
        good_freq_mask = ((freqs >= 5.0) & (freqs <= 15.0))
        weights[good_freq_mask] *= 2.2
        low_freq_mask = (freqs < 4.0)
        weights[low_freq_mask] *= 0.2
        mid_freq_mask = (freqs >= 4.0) & (freqs < 5.0)
        weights[mid_freq_mask] *= 0.6
    return weights

def smart_initial_guess_generator(freq_hz, target_n_range=(1.587, 1.817)):
    n_center = np.mean(target_n_range)
    freq_thz = freq_hz / 1e12
    if freq_thz < 2.0:
        n_guess = n_center + 0.015
    elif freq_thz > 12.0:
        n_guess = n_center - 0.010
    else:
        n_guess = n_center
    n_guess = np.clip(n_guess, target_n_range[0] + 0.005, target_n_range[1] - 0.005)
    k_guess = 0.012
    return [n_guess, k_guess]

def FP_f(n_tilde, f, N):
    r_sq = ((n_tilde - n_air) / (n_tilde + n_air)) ** 2
    x = r_sq * np.exp(-2j * n_tilde * (2 * np.pi * f * d / c))
    if np.abs(1 - x) < 1e-12:
        return float(N)
    return (1 - x**N) / (1 - x)

def error_function(n_k_pair, Tomega_measured, f, N, weight=1.0):
    n_tilde = complex(n_k_pair[0], n_k_pair[1])
    try:
        calculated_Tomega = (
            (4 * n_air * n_tilde / (n_air + n_tilde) ** 2)
            * np.exp(-1j * (n_tilde - n_air) * (2 * np.pi * f * d / c))
            * FP_f(n_tilde, f, N)
        )
        residual = (Tomega_measured - calculated_Tomega) * weight
        return np.array([residual.real, residual.imag])
    except (OverflowError, ZeroDivisionError, RuntimeWarning):
        return np.array([1e6, 1e6])

def global_error_function(n_k_pair, Tomega_measured, f_hz, N, weight=1.0):
    residuals = error_function(n_k_pair, Tomega_measured, f_hz, N, weight)
    return residuals[0]**2 + residuals[1]**2

def assess_convergence_quality(result, n_result, k_result, target_range=(1.587, 1.817)):
    if hasattr(result, 'fun'):
        residual_norm = result.fun if np.isscalar(result.fun) else np.sum(result.fun**2)
    else:
        residual_norm = result
    n_reasonable = 1.5 <= n_result <= 1.9
    k_reasonable = 0.0 <= k_result <= 0.4
    in_target_range = target_range[0] <= n_result <= target_range[1]
    high_quality = residual_norm < 0.03
    excellent_quality = residual_norm < 0.0005
    score = 0
    if n_reasonable: score += 1
    if k_reasonable: score += 1
    if in_target_range: score += 3
    if high_quality: score += 1
    return {
        'residual_norm': residual_norm,
        'physically_reasonable': n_reasonable and k_reasonable,
        'in_target_range': in_target_range,
        'high_quality': high_quality,
        'excellent_quality': excellent_quality,
        'overall_score': score,
        'priority_score': score + (3 if excellent_quality else 0)
    }

def global_optimization(Tomega, f_hz, N, weight, target_range=(1.587, 1.817)):
    hard_bounds = [(target_range[0], target_range[1]), (0.0, 0.25)]
    best_result = None
    best_quality = {'priority_score': -1, 'residual_norm': float('inf')}
    try:
        de_result = differential_evolution(
            global_error_function,
            bounds=hard_bounds,
            args=(Tomega, f_hz, N, weight),
            seed=42,
            maxiter=500,
            atol=1e-9,
            tol=1e-9,
            popsize=25,
            strategy='best1bin',
            workers=1
        )
        if de_result.success:
            quality = assess_convergence_quality(de_result.fun, de_result.x[0], de_result.x[1], target_range)
            if quality['priority_score'] > best_quality['priority_score']:
                best_result = de_result
                best_quality = quality
    except Exception:
        pass
    for attempt in range(3):
        try:
            initial_guess = smart_initial_guess_generator(f_hz, target_range)
            bh_result = basinhopping(
                global_error_function,
                initial_guess,
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'bounds': hard_bounds,
                    'args': (Tomega, f_hz, N, weight)
                },
                niter=200,
                T=0.03,
                stepsize=0.02,
                seed=42 + attempt
            )
            if bh_result.lowest_optimization_result.success:
                quality = assess_convergence_quality(
                    bh_result.fun, bh_result.x[0], bh_result.x[1], target_range
                )
                if quality['priority_score'] > best_quality['priority_score']:
                    best_result = bh_result
                    best_quality = quality
        except Exception:
            continue
    try:
        n_starts = 18
        n_range = np.linspace(target_range[0] + 0.002, target_range[1] - 0.002, n_starts)
        k_range = np.linspace(0.005, 0.035, 10)
        for n_start in n_range:
            for k_start in k_range:
                try:
                    local_result = minimize(
                        global_error_function,
                        [n_start, k_start],
                        args=(Tomega, f_hz, N, weight),
                        bounds=hard_bounds,
                        method='L-BFGS-B',
                        options={'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    if local_result.success:
                        quality = assess_convergence_quality(
                            local_result.fun, local_result.x[0], local_result.x[1], target_range
                        )
                        if quality['priority_score'] > best_quality['priority_score']:
                            best_result = local_result
                            best_quality = quality
                except Exception:
                    continue
    except Exception:
        pass
    return best_result, best_quality

def calculate_complex_refractive_index(freqs_thz, freqs_hz, Tomega, N):
    n_complex_values = []
    converged = []
    quality_metrics = []
    target_range = (1.587, 1.817)
    freq_weights = calculate_frequency_weights(freqs_thz, target_range=target_range)
    if hasattr(Tomega, 'imag') and np.any(np.imag(Tomega) != 0):
        phase_data = np.angle(Tomega)
        unwrapped_phase = preprocess_phase_data(phase_data)
        magnitude_data = np.abs(Tomega)
        Tomega_processed = magnitude_data * np.exp(1j * unwrapped_phase)
    else:
        Tomega_processed = Tomega
    for i, (f_thz, f_hz) in enumerate(zip(freqs_thz, freqs_hz)):
        current_weight = freq_weights[i]
        result, quality = global_optimization(
            Tomega_processed[i], f_hz, N, current_weight, target_range
        )
        if result is not None and quality['physically_reasonable']:
            if hasattr(result, 'x'):
                n_result, k_result = result.x
            else:
                n_result, k_result = result.lowest_optimization_result.x
            n_complex_values.append(complex(n_result, k_result))
            converged.append(True)
            quality_metrics.append(quality)
        else:
            n_complex_values.append(complex(np.nan, np.nan))
            converged.append(False)
            quality_metrics.append({'residual_norm': float('inf'), 'priority_score': 0})
    return np.array(n_complex_values), np.array(converged), quality_metrics

n_real = n_complex.real
k_imag = n_complex.imag

n_complex, conv_flags, quality_info = calculate_complex_refractive_index(freqs, freqs_hz, Tomega, N)

'''
Plotting the graph for n and k
'''

# Outlier Detection and Removal for n_real (using IQR) ---
n_real_converged = np.real(n_complex[conv_flags])
freqs_converged = freqs[conv_flags]

Q1_n = np.percentile(n_real_converged, 25)
Q3_n = np.percentile(n_real_converged, 75)
IQR_n = Q3_n - Q1_n

# Define outlier bounds for n
lower_bound_n = Q1_n - 1.5 * IQR_n
upper_bound_n = Q3_n + 1.5 * IQR_n

# Create a mask for non-outliers for n
non_outlier_mask_n = (n_real_converged >= lower_bound_n) & (n_real_converged <= upper_bound_n)

# Filter the data for n
n_real_filtered = n_real_converged[non_outlier_mask_n]
freqs_filtered_for_n = freqs_converged[non_outlier_mask_n]

print(f"Original converged points for n: {len(n_real_converged)}")
print(f"Points identified as n-outliers: {len(n_real_converged) - len(n_real_filtered)}")
print(f"Points remaining for n-fit: {len(n_real_filtered)}")


# Outlier Detection and Removal for k_imag (using Residuals from Initial Fit + IQR)
k_imag_converged = np.imag(n_complex[conv_flags]) # Ensure using n_complex from previous cell

# 1. Fit a 5th-degree polynomial to all converged k points initially
if len(k_imag_converged) > 5: 
    k_fit_initial = np.polyfit(freqs_converged, k_imag_converged, 5)
    k_curve_initial = np.poly1d(k_fit_initial)

    # 2. Calculate residuals
    residuals_k = k_imag_converged - k_curve_initial(freqs_converged)

    # 3. Use IQR on residuals to identify outliers
    Q1_res_k = np.percentile(residuals_k, 25)
    Q3_res_k = np.percentile(residuals_k, 75)
    IQR_res_k = Q3_res_k - Q1_res_k

    # Define outlier bounds for residuals
    lower_bound_res_k = Q1_res_k - 1.5 * IQR_res_k
    upper_bound_res_k = Q3_res_k + 1.5 * IQR_res_k

    # Create a mask for non-outliers based on residuals
    non_outlier_mask_k = (residuals_k >= lower_bound_res_k) & (residuals_k <= upper_bound_res_k)

    # Filter the data for k based on residual outliers
    k_imag_filtered = k_imag_converged[non_outlier_mask_k]
    freqs_filtered_for_k = freqs_converged[non_outlier_mask_k]

    print(f"\nOriginal converged points for k: {len(k_imag_converged)}")
    print(f"Points identified as k-outliers (based on residuals): {len(k_imag_converged) - len(k_imag_filtered)}")
    print(f"Points remaining for k-fit: {len(k_imag_filtered)}")

    # 4. Refit the 5th-degree polynomial to the filtered k data
    if len(k_imag_filtered) > 5: 
        k_fit = np.polyfit(freqs_filtered_for_k, k_imag_filtered, 5)
        k_curve = np.poly1d(k_fit)
        print(f"Polynomial fit coefficients for k (degree 5, after residual outlier removal): {k_fit}")
    else:
        k_curve = lambda x: np.nan 
        print("Not enough non-outlier points (after residual analysis) to fit a 5th degree polynomial to k.") 

else:
    k_curve = lambda x: np.nan 
    non_outlier_mask_k = np.zeros_like(k_imag_converged, dtype=bool) 
    freqs_filtered_for_k = np.array([])
    k_imag_filtered = np.array([])
    print("\nNot enough converged points to perform initial 5th degree polynomial fit for k outlier detection.") # Updated print


# Fitting for n (remains the same, using IQR on raw n data) 
if len(n_real_filtered) > 1:
    n_fit = np.polyfit(freqs_filtered_for_n, n_real_filtered, 1)
    n_line = np.poly1d(n_fit)
    print(f"\nLinear fit coefficients for n (after outlier removal): {n_fit}")
else:
    n_line = lambda x: np.nan 
    print("\nNot enough non-outlier points to fit a line to n.")


# Plotting 
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
# Plot all original calculated points for n 
plt.plot(freqs[conv_flags], n_real[conv_flags], 'o', label='Calculated Points', alpha=0.6, color='blue')
# Plot linear fit line for n (using the fit from filtered data)
plt.plot(freqs_converged, n_line(freqs_converged), '-', color='red', label='Linear Fit', linewidth=1.5)

plt.xlabel("Frequency (THz)")
plt.ylabel("Refractive Index (n)")
plt.grid(True) 
plt.ylim(0.0, 3.5) 
plt.legend()

plt.subplot(2, 1, 2) 
# Plot all original calculated points for k 
plt.plot(freqs[conv_flags], k_imag[conv_flags], 'o', label='Calculated Points', alpha=0.6, color='blue')
# Plot polynomial fit line for k (using the fit from filtered data)
plt.plot(freqs_converged, k_curve(freqs_converged), '-', color='red', label='Polynomial Fit (Degree 5)', linewidth=1.5) 

plt.xlabel("Frequency (THz)")
plt.ylabel("Imaginary part (k)")
plt.grid(True) 
plt.ylim(-0.02, 0.4)
plt.legend() 

plt.tight_layout()
plt.show()
