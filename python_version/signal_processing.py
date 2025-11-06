"""
Advanced Signal Processing and Filtering
Professional-grade signal conditioning for system identification
"""

import numpy as np
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SignalStats:
    """Statistical properties of signal"""
    mean: float
    std: float
    variance: float
    snr_db: float
    outlier_ratio: float


def preprocess_signal(
    data: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: Optional[float] = None,
    remove_outliers: bool = True,
    detrend: bool = True,
    smooth: bool = True,
    interpolate_missing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced signal preprocessing pipeline

    Args:
        data: Raw signal data
        timestamps: Time points
        sampling_rate: Desired sampling rate (Hz), None = auto
        remove_outliers: Remove statistical outliers
        detrend: Remove linear trend
        smooth: Apply smoothing filter
        interpolate_missing: Interpolate missing/NaN values

    Returns:
        Preprocessed signal and timestamps
    """

    # 1. Handle NaN and infinite values
    if interpolate_missing:
        data, timestamps = interpolate_missing_values(data, timestamps)

    # 2. Remove outliers using robust statistics
    if remove_outliers:
        data = remove_outliers_robust(data)

    # 3. Resample to uniform sampling rate
    if sampling_rate is not None:
        data, timestamps = resample_uniform(data, timestamps, sampling_rate)

    # 4. Detrend - remove DC offset and linear trend
    if detrend:
        data = signal.detrend(data, type='linear')

    # 5. Smooth using Savitzky-Golay filter (preserves features)
    if smooth:
        window_length = min(51, len(data) if len(data) % 2 == 1 else len(data) - 1)
        if window_length >= 5:
            data = signal.savgol_filter(data, window_length, 3)

    return data, timestamps


def remove_outliers_robust(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Remove outliers using Modified Z-score (MAD-based)
    More robust than standard deviation for non-normal distributions
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad == 0:
        return data

    # Modified Z-score
    modified_z_scores = 0.6745 * (data - median) / mad

    # Replace outliers with median
    outlier_mask = np.abs(modified_z_scores) > threshold
    cleaned_data = data.copy()
    cleaned_data[outlier_mask] = median

    return cleaned_data


def interpolate_missing_values(
    data: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate NaN and infinite values"""

    # Find valid points
    valid_mask = np.isfinite(data)

    if not np.any(valid_mask):
        raise ValueError("All data points are invalid")

    if np.all(valid_mask):
        return data, timestamps

    # Interpolate invalid points
    f = interpolate.interp1d(
        timestamps[valid_mask],
        data[valid_mask],
        kind='cubic',
        bounds_error=False,
        fill_value='extrapolate'
    )

    data_clean = f(timestamps)

    return data_clean, timestamps


def resample_uniform(
    data: np.ndarray,
    timestamps: np.ndarray,
    target_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to uniform sampling rate"""

    if len(data) < 2:
        return data, timestamps

    # Create uniform time grid
    t_start = timestamps[0]
    t_end = timestamps[-1]
    dt = 1.0 / target_rate

    new_timestamps = np.arange(t_start, t_end, dt)

    # Interpolate to new grid
    f = interpolate.interp1d(
        timestamps,
        data,
        kind='cubic',
        bounds_error=False,
        fill_value='extrapolate'
    )

    new_data = f(new_timestamps)

    return new_data, new_timestamps


def apply_butterworth_filter(
    data: np.ndarray,
    sampling_rate: float,
    cutoff_freq: float,
    order: int = 4,
    filter_type: str = 'low'
) -> np.ndarray:
    """
    Apply Butterworth filter

    Args:
        filter_type: 'low', 'high', 'band', or 'notch'
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    if filter_type == 'low':
        b, a = signal.butter(order, normalized_cutoff, btype='low')
    elif filter_type == 'high':
        b, a = signal.butter(order, normalized_cutoff, btype='high')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply forward-backward filter (zero phase)
    filtered = signal.filtfilt(b, a, data)

    return filtered


def apply_kalman_filter(
    measurements: np.ndarray,
    process_variance: float = 1e-5,
    measurement_variance: float = 0.1
) -> np.ndarray:
    """
    Simple 1D Kalman filter for smoothing
    """
    n = len(measurements)

    # Initialize
    x_hat = np.zeros(n)  # Estimate
    P = np.zeros(n)      # Error covariance

    x_hat[0] = measurements[0]
    P[0] = 1.0

    Q = process_variance       # Process noise
    R = measurement_variance   # Measurement noise

    for k in range(1, n):
        # Prediction
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q

        # Update
        K = P_minus / (P_minus + R)  # Kalman gain
        x_hat[k] = x_hat_minus + K * (measurements[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus

    return x_hat


def compute_signal_stats(data: np.ndarray) -> SignalStats:
    """Compute comprehensive signal statistics"""

    mean = np.mean(data)
    std = np.std(data)
    variance = np.var(data)

    # SNR estimation
    signal_power = np.mean(data ** 2)
    noise = data - signal.savgol_filter(data, 51, 3) if len(data) > 51 else data - np.mean(data)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Outlier detection
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    outliers = np.abs(0.6745 * (data - median) / (mad + 1e-10)) > 3.5
    outlier_ratio = np.sum(outliers) / len(data)

    return SignalStats(
        mean=mean,
        std=std,
        variance=variance,
        snr_db=snr_db,
        outlier_ratio=outlier_ratio
    )


def frequency_analysis(
    data: np.ndarray,
    sampling_rate: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform frequency domain analysis

    Returns:
        frequencies, magnitudes, dominant_frequency
    """
    n = len(data)

    # Apply window to reduce spectral leakage
    window = signal.windows.hann(n)
    windowed_data = data * window

    # FFT
    fft_vals = fft(windowed_data)
    freqs = fftfreq(n, 1/sampling_rate)

    # Take positive frequencies only
    positive_freq_idx = freqs > 0
    freqs = freqs[positive_freq_idx]
    magnitudes = np.abs(fft_vals[positive_freq_idx])

    # Find dominant frequency
    dominant_idx = np.argmax(magnitudes)
    dominant_freq = freqs[dominant_idx]

    return freqs, magnitudes, dominant_freq


def compute_power_spectral_density(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method
    """
    if nperseg is None:
        nperseg = min(256, len(data) // 4)

    freqs, psd = signal.welch(
        data,
        fs=sampling_rate,
        nperseg=nperseg,
        scaling='density'
    )

    return freqs, psd


def detect_oscillations_advanced(
    data: np.ndarray,
    timestamps: np.ndarray,
    min_prominence: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Advanced oscillation detection using peak finding

    Returns:
        peak_indices, peak_values, period, amplitude
    """
    # Find peaks
    peaks, properties = signal.find_peaks(
        data,
        prominence=min_prominence * np.std(data),
        distance=5
    )

    if len(peaks) < 2:
        return np.array([]), np.array([]), 0.0, 0.0

    # Calculate periods between peaks
    peak_times = timestamps[peaks]
    periods = np.diff(peak_times)

    mean_period = np.mean(periods)

    # Calculate amplitude
    peak_values = data[peaks]
    amplitude = np.mean(np.abs(peak_values))

    return peaks, peak_values, mean_period, amplitude


def segment_signal_by_activity(
    data: np.ndarray,
    timestamps: np.ndarray,
    threshold_factor: float = 0.1
) -> List[Tuple[int, int]]:
    """
    Segment signal into active and inactive regions
    Returns list of (start_idx, end_idx) for active segments
    """
    # Compute moving variance
    window_size = min(50, len(data) // 10)

    variance = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2)
        variance[i] = np.var(data[start:end])

    # Threshold for activity
    threshold = threshold_factor * np.max(variance)
    active = variance > threshold

    # Find continuous active segments
    segments = []
    in_segment = False
    start_idx = 0

    for i in range(len(active)):
        if active[i] and not in_segment:
            start_idx = i
            in_segment = True
        elif not active[i] and in_segment:
            segments.append((start_idx, i))
            in_segment = False

    if in_segment:
        segments.append((start_idx, len(active)))

    return segments


def compute_derivative(
    data: np.ndarray,
    timestamps: np.ndarray,
    method: str = 'gradient'
) -> np.ndarray:
    """
    Compute derivative with various methods

    Methods:
        - 'gradient': Central differences
        - 'savgol': Savitzky-Golay derivative
        - 'spline': Spline derivative
    """
    if method == 'gradient':
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # Pad
        derivative = np.gradient(data, timestamps)

    elif method == 'savgol':
        window_length = min(51, len(data) if len(data) % 2 == 1 else len(data) - 1)
        if window_length < 5:
            return np.gradient(data, timestamps)
        derivative = signal.savgol_filter(data, window_length, 3, deriv=1, delta=np.mean(np.diff(timestamps)))

    elif method == 'spline':
        spline = interpolate.UnivariateSpline(timestamps, data, s=0, k=3)
        derivative = spline.derivative()(timestamps)

    else:
        raise ValueError(f"Unknown method: {method}")

    return derivative


def compute_integral(
    data: np.ndarray,
    timestamps: np.ndarray,
    method: str = 'trapz'
) -> np.ndarray:
    """
    Compute cumulative integral

    Methods:
        - 'trapz': Trapezoidal rule
        - 'simpson': Simpson's rule (requires odd number of points)
    """
    if method == 'trapz':
        integral = np.zeros_like(data)
        for i in range(1, len(data)):
            dt = timestamps[i] - timestamps[i-1]
            integral[i] = integral[i-1] + 0.5 * (data[i] + data[i-1]) * dt

    elif method == 'simpson':
        from scipy.integrate import cumulative_trapezoid
        integral = cumulative_trapezoid(data, timestamps, initial=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return integral
