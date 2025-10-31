"""
PID coefficient calculation algorithms
Implements Ziegler-Nichols, Relay Method, and System Identification
"""

import numpy as np
from typing import List
from models import FlightDataPoint, PIDCoefficients, PIDResults


def calculate_pid_coefficients(
    data: List[FlightDataPoint],
    axis: str,
    method: str
) -> PIDResults:
    """Main function to calculate PID coefficients"""

    results = PIDResults(method=method)

    if method == 'ziegler-nichols':
        if axis in ['all', 'roll']:
            results.roll = calculate_ziegler_nichols_for_axis(data, 'roll')
            results.analysis_notes.append("Roll: Using Ziegler-Nichols tuning")

        if axis in ['all', 'pitch']:
            results.pitch = calculate_ziegler_nichols_for_axis(data, 'pitch')
            results.analysis_notes.append("Pitch: Using Ziegler-Nichols tuning")

        if axis in ['all', 'yaw']:
            results.yaw = calculate_ziegler_nichols_for_axis(data, 'yaw')
            results.analysis_notes.append("Yaw: Using Ziegler-Nichols tuning")

        if axis in ['all', 'alt']:
            results.altitude = calculate_ziegler_nichols_for_axis(data, 'alt')
            results.analysis_notes.append("Altitude: Using Ziegler-Nichols tuning")

    elif method == 'relay':
        if axis in ['all', 'roll']:
            results.roll = calculate_relay_method_for_axis(data, 'roll')

        if axis in ['all', 'pitch']:
            results.pitch = calculate_relay_method_for_axis(data, 'pitch')

        if axis in ['all', 'yaw']:
            results.yaw = calculate_relay_method_for_axis(data, 'yaw')

        if axis in ['all', 'alt']:
            results.altitude = calculate_relay_method_for_axis(data, 'alt')

    elif method == 'manual':
        if axis in ['all', 'roll']:
            results.roll = calculate_system_id_for_axis(data, 'roll')

        if axis in ['all', 'pitch']:
            results.pitch = calculate_system_id_for_axis(data, 'pitch')

        if axis in ['all', 'yaw']:
            results.yaw = calculate_system_id_for_axis(data, 'yaw')

        if axis in ['all', 'alt']:
            results.altitude = calculate_system_id_for_axis(data, 'alt')

    else:
        raise ValueError(f"Unknown method: {method}")

    return results


def calculate_ziegler_nichols_for_axis(
    data: List[FlightDataPoint],
    axis: str
) -> PIDCoefficients:
    """Calculate PID using Ziegler-Nichols method"""

    # Extract error signal
    errors = []
    for point in data:
        if axis == 'roll':
            errors.append(point.roll_error())
        elif axis == 'pitch':
            errors.append(point.pitch_error())
        elif axis == 'yaw':
            errors.append(point.yaw_error())
        elif axis == 'alt':
            errors.append(point.alt_error())

    errors = np.array(errors)

    if len(errors) < 10:
        raise ValueError(f"Insufficient data for axis: {axis}")

    # Analyze oscillations
    ku, tu = analyze_oscillations(errors, data)

    # Ziegler-Nichols formulas (PID controller)
    p = 0.6 * ku
    i = 2.0 * p / tu
    d = p * tu / 8.0

    return PIDCoefficients(p=p, i=i, d=d)


def calculate_relay_method_for_axis(
    data: List[FlightDataPoint],
    axis: str
) -> PIDCoefficients:
    """Calculate PID using relay feedback method"""

    errors = []
    outputs = []

    for point in data:
        if axis == 'roll':
            errors.append(point.roll_error())
            outputs.append(point.roll_output)
        elif axis == 'pitch':
            errors.append(point.pitch_error())
            outputs.append(point.pitch_output)
        elif axis == 'yaw':
            errors.append(point.yaw_error())
            outputs.append(point.yaw_output)
        elif axis == 'alt':
            errors.append(point.alt_error())
            outputs.append(point.throttle)

    errors = np.array(errors)
    outputs = np.array(outputs)

    # Detect limit cycles
    period, amp_output, amp_process = detect_limit_cycle(errors, outputs, data)

    # Calculate ultimate gain
    ku = (4.0 * amp_output) / (np.pi * amp_process)
    tu = period

    # Modified Ziegler-Nichols for relay method (more conservative)
    p = 0.45 * ku
    i = 1.2 * p / tu
    d = p * tu / 10.0

    return PIDCoefficients(p=p, i=i, d=d)


def calculate_system_id_for_axis(
    data: List[FlightDataPoint],
    axis: str
) -> PIDCoefficients:
    """Calculate PID using system identification"""

    errors = []
    outputs = []

    for point in data:
        if axis == 'roll':
            errors.append(point.roll_error())
            outputs.append(point.roll_output)
        elif axis == 'pitch':
            errors.append(point.pitch_error())
            outputs.append(point.pitch_output)
        elif axis == 'yaw':
            errors.append(point.yaw_error())
            outputs.append(point.yaw_output)
        elif axis == 'alt':
            errors.append(point.alt_error())
            outputs.append(point.throttle)

    errors = np.array(errors)
    outputs = np.array(outputs)

    # Least squares estimation
    coeffs = least_squares_pid_estimate(errors, outputs)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(errors)

    # Adjust based on metrics
    p = coeffs.p * (1.0 - 0.1 * min(metrics['overshoot'], 1.0))
    i = coeffs.i * (2.0 / (1.0 + metrics['settling_time']))
    d = coeffs.d * (1.0 + 0.1 * metrics['oscillation_count'])

    return PIDCoefficients(p=p, i=i, d=d)


def analyze_oscillations(errors: np.ndarray, data: List[FlightDataPoint]):
    """Analyze oscillations to find Ku and Tu"""

    # Find zero crossings
    crossings = []
    for i in range(1, len(errors)):
        if errors[i - 1] * errors[i] < 0:
            crossings.append(i)

    if len(crossings) < 4:
        # Not enough oscillations, use conservative defaults
        return 0.5, 1.0

    # Calculate periods
    periods = []
    for i in range(2, len(crossings)):
        t1 = data[crossings[i - 2]].timestamp
        t2 = data[crossings[i]].timestamp
        periods.append(t2 - t1)

    tu = np.mean(periods) if periods else 1.0

    # Estimate ultimate gain
    max_error = np.max(np.abs(errors))
    ku = 1.0 / max_error if max_error > 0.01 else 1.0

    return ku, tu


def detect_limit_cycle(errors: np.ndarray, outputs: np.ndarray, data: List[FlightDataPoint]):
    """Detect limit cycle characteristics"""

    # Find peaks in error signal
    peaks = []
    for i in range(1, len(errors) - 1):
        if errors[i] > errors[i - 1] and errors[i] > errors[i + 1] and abs(errors[i]) > 0.01:
            peaks.append((i, errors[i]))

    if len(peaks) < 2:
        return 1.0, 0.1, 0.1

    # Calculate period
    periods = []
    for i in range(1, len(peaks)):
        t1 = data[peaks[i - 1][0]].timestamp
        t2 = data[peaks[i][0]].timestamp
        periods.append(t2 - t1)

    period = np.mean(periods) if periods else 1.0

    # Calculate amplitudes
    amp_process = np.mean([abs(p[1]) for p in peaks])
    amp_output = np.max(np.abs(outputs))

    return period, amp_output, amp_process


def least_squares_pid_estimate(errors: np.ndarray, outputs: np.ndarray) -> PIDCoefficients:
    """Least squares estimation of PID parameters"""

    n = min(len(errors), len(outputs))
    if n < 3:
        return PIDCoefficients(p=0.5, i=0.1, d=0.05)

    # Build regression matrix
    # output = P*error + I*sum(error) + D*diff(error)
    A = []
    b = []

    error_sum = 0.0
    for i in range(1, n - 1):
        error = errors[i]
        error_sum += error
        error_diff = errors[i] - errors[i - 1] if i > 0 else 0.0

        A.append([error, error_sum, error_diff])
        b.append(outputs[i])

    A = np.array(A)
    b = np.array(b)

    if len(A) < 3:
        return PIDCoefficients(p=0.5, i=0.1, d=0.05)

    try:
        # Solve using least squares
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Normalize and bound coefficients
        p = abs(x[0])
        p = max(0.1, min(p, 2.0))

        i = abs(x[1])
        i = max(0.01, min(i, 1.0))

        d = abs(x[2])
        d = max(0.01, min(d, 0.5))

        return PIDCoefficients(p=p, i=i, d=d)

    except:
        return PIDCoefficients(p=0.5, i=0.1, d=0.05)


def calculate_performance_metrics(errors: np.ndarray) -> dict:
    """Calculate performance metrics from error signal"""

    # Overshoot
    max_error = np.max(np.abs(errors))
    min_error = np.min(np.abs(errors))
    overshoot = max(max_error, min_error) - 1.0
    overshoot = max(overshoot, 0.0)

    # Settling time (rough estimate)
    threshold = max_error * 0.05
    settling_idx = len(errors)
    for i in range(len(errors) - 1, -1, -1):
        if abs(errors[i]) > threshold:
            settling_idx = i
            break

    settling_time = settling_idx / len(errors)

    # Count oscillations
    oscillation_count = 0
    for i in range(1, len(errors)):
        if errors[i - 1] * errors[i] < 0:
            oscillation_count += 1

    return {
        'overshoot': overshoot,
        'settling_time': settling_time,
        'oscillation_count': oscillation_count
    }
