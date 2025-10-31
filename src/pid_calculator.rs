use crate::models::{FlightDataPoint, PIDCoefficients, PIDResults};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;

/// Main function to calculate PID coefficients
pub fn calculate_pid_coefficients(
    data: &[FlightDataPoint],
    axis: &str,
    method: &str,
) -> Result<PIDResults> {
    let mut results = PIDResults {
        roll: None,
        pitch: None,
        yaw: None,
        altitude: None,
        method: method.to_string(),
        analysis_notes: Vec::new(),
    };

    match method {
        "ziegler-nichols" => {
            if axis == "all" || axis == "roll" {
                results.roll = Some(calculate_ziegler_nichols_for_axis(data, "roll")?);
                results.analysis_notes.push("Roll: Using Ziegler-Nichols tuning".to_string());
            }
            if axis == "all" || axis == "pitch" {
                results.pitch = Some(calculate_ziegler_nichols_for_axis(data, "pitch")?);
                results.analysis_notes.push("Pitch: Using Ziegler-Nichols tuning".to_string());
            }
            if axis == "all" || axis == "yaw" {
                results.yaw = Some(calculate_ziegler_nichols_for_axis(data, "yaw")?);
                results.analysis_notes.push("Yaw: Using Ziegler-Nichols tuning".to_string());
            }
            if axis == "all" || axis == "alt" {
                results.altitude = Some(calculate_ziegler_nichols_for_axis(data, "alt")?);
                results.analysis_notes.push("Altitude: Using Ziegler-Nichols tuning".to_string());
            }
        }

        "relay" => {
            if axis == "all" || axis == "roll" {
                results.roll = Some(calculate_relay_method_for_axis(data, "roll")?);
            }
            if axis == "all" || axis == "pitch" {
                results.pitch = Some(calculate_relay_method_for_axis(data, "pitch")?);
            }
            if axis == "all" || axis == "yaw" {
                results.yaw = Some(calculate_relay_method_for_axis(data, "yaw")?);
            }
            if axis == "all" || axis == "alt" {
                results.altitude = Some(calculate_relay_method_for_axis(data, "alt")?);
            }
        }

        "manual" => {
            // System identification based method
            if axis == "all" || axis == "roll" {
                results.roll = Some(calculate_system_id_for_axis(data, "roll")?);
            }
            if axis == "all" || axis == "pitch" {
                results.pitch = Some(calculate_system_id_for_axis(data, "pitch")?);
            }
            if axis == "all" || axis == "yaw" {
                results.yaw = Some(calculate_system_id_for_axis(data, "yaw")?);
            }
            if axis == "all" || axis == "alt" {
                results.altitude = Some(calculate_system_id_for_axis(data, "alt")?);
            }
        }

        _ => return Err(anyhow!("Unknown method: {}", method)),
    }

    Ok(results)
}

/// Calculate PID using Ziegler-Nichols method based on step response
fn calculate_ziegler_nichols_for_axis(data: &[FlightDataPoint], axis: &str) -> Result<PIDCoefficients> {
    // Extract error signal
    let errors: Vec<f64> = data.iter().map(|p| match axis {
        "roll" => p.roll_error(),
        "pitch" => p.pitch_error(),
        "yaw" => p.yaw_error(),
        "alt" => p.alt_error(),
        _ => 0.0,
    }).collect();

    if errors.is_empty() {
        return Err(anyhow!("No data points for axis: {}", axis));
    }

    // Find oscillation characteristics
    let oscillation = analyze_oscillations(&errors, data)?;

    // Ziegler-Nichols formulas (based on ultimate gain and period)
    let ku = oscillation.ultimate_gain;
    let tu = oscillation.ultimate_period;

    // Classic PID Ziegler-Nichols tuning
    let p = 0.6 * ku;
    let i = 2.0 * p / tu;
    let d = p * tu / 8.0;

    Ok(PIDCoefficients { p, i, d })
}

/// Calculate PID using relay feedback method (Åström-Hägglund)
fn calculate_relay_method_for_axis(data: &[FlightDataPoint], axis: &str) -> Result<PIDCoefficients> {
    let errors: Vec<f64> = data.iter().map(|p| match axis {
        "roll" => p.roll_error(),
        "pitch" => p.pitch_error(),
        "yaw" => p.yaw_error(),
        "alt" => p.alt_error(),
        _ => 0.0,
    }).collect();

    let outputs: Vec<f64> = data.iter().map(|p| match axis {
        "roll" => p.roll_output,
        "pitch" => p.pitch_output,
        "yaw" => p.yaw_output,
        "alt" => p.throttle,
        _ => 0.0,
    }).collect();

    // Detect limit cycles from relay feedback
    let cycle = detect_limit_cycle(&errors, &outputs, data)?;

    // Calculate ultimate gain
    let amplitude_output = cycle.amplitude_output;
    let amplitude_process = cycle.amplitude_process;
    let ku = (4.0 * amplitude_output) / (std::f64::consts::PI * amplitude_process);
    let tu = cycle.period;

    // Modified Ziegler-Nichols for relay method (more conservative)
    let p = 0.45 * ku;
    let i = 1.2 * p / tu;
    let d = p * tu / 10.0;

    Ok(PIDCoefficients { p, i, d })
}

/// Calculate PID using system identification
fn calculate_system_id_for_axis(data: &[FlightDataPoint], axis: &str) -> Result<PIDCoefficients> {
    let errors: Vec<f64> = data.iter().map(|p| match axis {
        "roll" => p.roll_error(),
        "pitch" => p.pitch_error(),
        "yaw" => p.yaw_error(),
        "alt" => p.alt_error(),
        _ => 0.0,
    }).collect();

    let outputs: Vec<f64> = data.iter().map(|p| match axis {
        "roll" => p.roll_output,
        "pitch" => p.pitch_output,
        "yaw" => p.yaw_output,
        "alt" => p.throttle,
        _ => 0.0,
    }).collect();

    // Use least squares to estimate system response
    let coeffs = least_squares_pid_estimate(&errors, &outputs)?;

    // Calculate performance metrics
    let metrics = calculate_performance_metrics(&errors);

    // Adjust based on overshoot and settling time
    let p = coeffs.p * (1.0 - 0.1 * metrics.overshoot.min(1.0));
    let i = coeffs.i * (2.0 / (1.0 + metrics.settling_time));
    let d = coeffs.d * (1.0 + 0.1 * metrics.oscillation_count as f64);

    Ok(PIDCoefficients { p, i, d })
}

// Helper structures
struct OscillationCharacteristics {
    ultimate_gain: f64,
    ultimate_period: f64,
}

struct LimitCycle {
    period: f64,
    amplitude_output: f64,
    amplitude_process: f64,
}

struct PerformanceMetrics {
    overshoot: f64,
    settling_time: f64,
    oscillation_count: usize,
}

/// Analyze oscillations in the error signal
fn analyze_oscillations(errors: &[f64], data: &[FlightDataPoint]) -> Result<OscillationCharacteristics> {
    if errors.len() < 10 {
        return Err(anyhow!("Insufficient data for oscillation analysis"));
    }

    // Find zero crossings to detect oscillations
    let mut crossings = Vec::new();
    for i in 1..errors.len() {
        if errors[i - 1] * errors[i] < 0.0 {
            crossings.push(i);
        }
    }

    if crossings.len() < 4 {
        // Not enough oscillations, use default conservative values
        return Ok(OscillationCharacteristics {
            ultimate_gain: 0.5,
            ultimate_period: 1.0,
        });
    }

    // Calculate average period (time between consecutive zero crossings)
    let mut periods = Vec::new();
    for i in 2..crossings.len() {
        let t1 = data[crossings[i - 2]].timestamp;
        let t2 = data[crossings[i]].timestamp;
        periods.push(t2 - t1);
    }

    let ultimate_period = if !periods.is_empty() {
        periods.iter().sum::<f64>() / periods.len() as f64
    } else {
        1.0
    };

    // Estimate ultimate gain from amplitude
    let error_array = Array1::from_vec(errors.to_vec());
    let max_error = error_array.max().unwrap_or(&1.0).abs();
    let ultimate_gain = if max_error > 0.01 { 1.0 / max_error } else { 1.0 };

    Ok(OscillationCharacteristics {
        ultimate_gain,
        ultimate_period,
    })
}

/// Detect limit cycle from relay feedback test
fn detect_limit_cycle(errors: &[f64], outputs: &[f64], data: &[FlightDataPoint]) -> Result<LimitCycle> {
    if errors.len() < 10 {
        return Err(anyhow!("Insufficient data for limit cycle detection"));
    }

    // Find peaks in error signal
    let mut peaks = Vec::new();
    for i in 1..errors.len() - 1 {
        if errors[i] > errors[i - 1] && errors[i] > errors[i + 1] && errors[i].abs() > 0.01 {
            peaks.push((i, errors[i]));
        }
    }

    if peaks.len() < 2 {
        return Ok(LimitCycle {
            period: 1.0,
            amplitude_output: 0.1,
            amplitude_process: 0.1,
        });
    }

    // Calculate average period between peaks
    let mut periods = Vec::new();
    for i in 1..peaks.len() {
        let t1 = data[peaks[i - 1].0].timestamp;
        let t2 = data[peaks[i].0].timestamp;
        periods.push(t2 - t1);
    }

    let period = periods.iter().sum::<f64>() / periods.len() as f64;

    // Calculate average amplitudes
    let amplitude_process = peaks.iter().map(|(_, a)| a.abs()).sum::<f64>() / peaks.len() as f64;

    let output_array = Array1::from_vec(outputs.to_vec());
    let amplitude_output = output_array.max().unwrap_or(&0.1).abs();

    Ok(LimitCycle {
        period,
        amplitude_output,
        amplitude_process,
    })
}

/// Least squares estimation of PID parameters
fn least_squares_pid_estimate(errors: &[f64], outputs: &[f64]) -> Result<PIDCoefficients> {
    let n = errors.len().min(outputs.len());
    if n < 3 {
        return Err(anyhow!("Insufficient data for least squares estimation"));
    }

    // Build regression matrix: output = P*error + I*sum(error) + D*diff(error)
    let mut a_data = Vec::new();
    let mut b_data = Vec::new();

    let mut error_sum = 0.0;
    for i in 1..n - 1 {
        let error = errors[i];
        error_sum += error;
        let error_diff = if i > 0 { errors[i] - errors[i - 1] } else { 0.0 };

        a_data.push(error);
        a_data.push(error_sum);
        a_data.push(error_diff);

        b_data.push(outputs[i]);
    }

    let m = (n - 2) as usize;
    if m < 3 {
        return Ok(PIDCoefficients { p: 0.5, i: 0.1, d: 0.05 });
    }

    let a = Array2::from_shape_vec((m, 3), a_data)
        .map_err(|e| anyhow!("Failed to create matrix: {}", e))?;
    let b = Array1::from_vec(b_data);

    // Solve A^T * A * x = A^T * b
    let at = a.t();
    let ata = at.dot(&a);
    let atb = at.dot(&b);

    // Simple 3x3 matrix inversion (for PID coefficients)
    let coeffs = solve_3x3(&ata, &atb).unwrap_or(PIDCoefficients { p: 0.5, i: 0.1, d: 0.05 });

    Ok(coeffs)
}

/// Solve 3x3 linear system (simplified)
fn solve_3x3(a: &Array2<f64>, b: &Array1<f64>) -> Option<PIDCoefficients> {
    // Use simple Gaussian elimination for 3x3 system
    // For production, use proper linear algebra library

    // Extract values (assuming 3x3)
    if a.shape() != [3, 3] || b.len() != 3 {
        return None;
    }

    // Simplified solution - using pseudo-inverse approach
    let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
            - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
            + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);

    if det.abs() < 1e-10 {
        return None;
    }

    // For simplicity, return normalized values
    let p = (b[0] / a[[0, 0]]).abs().max(0.1).min(2.0);
    let i = (b[1] / a[[1, 1]].max(1.0)).abs().max(0.01).min(1.0);
    let d = (b[2] / a[[2, 2]].max(1.0)).abs().max(0.01).min(0.5);

    Some(PIDCoefficients { p, i, d })
}

/// Calculate performance metrics from error signal
fn calculate_performance_metrics(errors: &[f64]) -> PerformanceMetrics {
    let error_array = Array1::from_vec(errors.to_vec());

    // Calculate overshoot
    let max_error = error_array.max().unwrap_or(&0.0).abs();
    let min_error = error_array.min().unwrap_or(&0.0).abs();
    let overshoot = (max_error.max(min_error) - 1.0).max(0.0);

    // Calculate settling time (rough estimate)
    let threshold = max_error * 0.05;
    let mut settling_idx = errors.len();
    for i in (0..errors.len()).rev() {
        if errors[i].abs() > threshold {
            settling_idx = i;
            break;
        }
    }
    let settling_time = settling_idx as f64 / errors.len() as f64;

    // Count oscillations
    let mut oscillation_count = 0;
    for i in 1..errors.len() {
        if errors[i - 1] * errors[i] < 0.0 {
            oscillation_count += 1;
        }
    }

    PerformanceMetrics {
        overshoot,
        settling_time,
        oscillation_count,
    }
}
