use crate::models::FlightDataPoint;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse Ardupilot log file (supports .log and .csv formats)
pub fn parse_log(path: &Path) -> Result<Vec<FlightDataPoint>> {
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match extension {
        "csv" => parse_csv_log(path),
        "log" | "txt" => parse_text_log(path),
        _ => parse_csv_log(path), // Try CSV by default
    }
}

/// Parse CSV format log (exported from MAVExplorer or Mission Planner)
fn parse_csv_log(path: &Path) -> Result<Vec<FlightDataPoint>> {
    let mut reader = csv::Reader::from_path(path)
        .context("Failed to open CSV file")?;

    let mut data_points = Vec::new();

    // Get headers
    let headers = reader.headers()?.clone();

    for result in reader.records() {
        let record = result?;

        // Try to extract relevant fields
        let timestamp = get_field_value(&record, &headers, &["timestamp", "time", "TimeUS"])
            .unwrap_or(0.0) / 1_000_000.0; // Convert microseconds to seconds

        let roll_desired = get_field_value(&record, &headers, &["DesRoll", "RollIn", "roll_des"]).unwrap_or(0.0);
        let roll_actual = get_field_value(&record, &headers, &["Roll", "roll"]).unwrap_or(0.0);

        let pitch_desired = get_field_value(&record, &headers, &["DesPitch", "PitchIn", "pitch_des"]).unwrap_or(0.0);
        let pitch_actual = get_field_value(&record, &headers, &["Pitch", "pitch"]).unwrap_or(0.0);

        let yaw_desired = get_field_value(&record, &headers, &["DesYaw", "YawIn", "yaw_des"]).unwrap_or(0.0);
        let yaw_actual = get_field_value(&record, &headers, &["Yaw", "yaw"]).unwrap_or(0.0);

        let alt_desired = get_field_value(&record, &headers, &["DAlt", "alt_des", "PosD"]).unwrap_or(0.0);
        let alt_actual = get_field_value(&record, &headers, &["Alt", "alt", "PosD"]).unwrap_or(0.0);

        let roll_rate = get_field_value(&record, &headers, &["GyrX", "rollRate", "gyro_x"]).unwrap_or(0.0);
        let pitch_rate = get_field_value(&record, &headers, &["GyrY", "pitchRate", "gyro_y"]).unwrap_or(0.0);
        let yaw_rate = get_field_value(&record, &headers, &["GyrZ", "yawRate", "gyro_z"]).unwrap_or(0.0);

        let throttle = get_field_value(&record, &headers, &["ThO", "ThrOut", "throttle"]).unwrap_or(0.0);
        let roll_output = get_field_value(&record, &headers, &["ROut", "rollOut"]).unwrap_or(0.0);
        let pitch_output = get_field_value(&record, &headers, &["POut", "pitchOut"]).unwrap_or(0.0);
        let yaw_output = get_field_value(&record, &headers, &["YOut", "yawOut"]).unwrap_or(0.0);

        data_points.push(FlightDataPoint {
            timestamp,
            roll_desired,
            roll_actual,
            pitch_desired,
            pitch_actual,
            yaw_desired,
            yaw_actual,
            alt_desired,
            alt_actual,
            roll_rate,
            pitch_rate,
            yaw_rate,
            throttle,
            roll_output,
            pitch_output,
            yaw_output,
        });
    }

    Ok(data_points)
}

/// Parse text format log (native .log format)
fn parse_text_log(path: &Path) -> Result<Vec<FlightDataPoint>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data_points = Vec::new();
    let mut timestamp = 0.0;

    // Temporary storage for different message types
    let mut current_att = (0.0, 0.0, 0.0);      // roll, pitch, yaw
    let mut current_att_des = (0.0, 0.0, 0.0);  // desired
    let mut current_rate = (0.0, 0.0, 0.0);     // rates
    let mut current_alt = (0.0, 0.0);           // desired, actual
    let mut current_output = (0.0, 0.0, 0.0, 0.0); // roll, pitch, yaw, throttle

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        if parts.is_empty() {
            continue;
        }

        let msg_type = parts[0];

        match msg_type {
            "ATT" if parts.len() > 4 => {
                // Attitude: TimeUS, DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw
                if let (Some(t), Some(roll), Some(pitch), Some(yaw)) = (
                    parts.get(1).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(3).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(5).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(7).and_then(|s| s.parse::<f64>().ok()),
                ) {
                    timestamp = t / 1_000_000.0;
                    current_att = (roll, pitch, yaw);

                    if let (Some(des_roll), Some(des_pitch), Some(des_yaw)) = (
                        parts.get(2).and_then(|s| s.parse::<f64>().ok()),
                        parts.get(4).and_then(|s| s.parse::<f64>().ok()),
                        parts.get(6).and_then(|s| s.parse::<f64>().ok()),
                    ) {
                        current_att_des = (des_roll, des_pitch, des_yaw);
                    }
                }
            }

            "RATE" if parts.len() > 4 => {
                // Rate: TimeUS, RDes, R, PDes, P, YDes, Y
                if let (Some(roll_r), Some(pitch_r), Some(yaw_r)) = (
                    parts.get(3).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(5).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(7).and_then(|s| s.parse::<f64>().ok()),
                ) {
                    current_rate = (roll_r, pitch_r, yaw_r);
                }
            }

            "CTUN" if parts.len() > 5 => {
                // Control tuning: includes altitude info
                if let (Some(alt_des), Some(alt)) = (
                    parts.get(2).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(3).and_then(|s| s.parse::<f64>().ok()),
                ) {
                    current_alt = (alt_des, alt);
                }
            }

            "RCOU" if parts.len() > 4 => {
                // RC output
                if let (Some(r), Some(p), Some(y), Some(t)) = (
                    parts.get(2).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(3).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(4).and_then(|s| s.parse::<f64>().ok()),
                    parts.get(5).and_then(|s| s.parse::<f64>().ok()),
                ) {
                    current_output = (r, p, y, t);

                    // Create a data point when we have output data
                    data_points.push(FlightDataPoint {
                        timestamp,
                        roll_desired: current_att_des.0,
                        roll_actual: current_att.0,
                        pitch_desired: current_att_des.1,
                        pitch_actual: current_att.1,
                        yaw_desired: current_att_des.2,
                        yaw_actual: current_att.2,
                        alt_desired: current_alt.0,
                        alt_actual: current_alt.1,
                        roll_rate: current_rate.0,
                        pitch_rate: current_rate.1,
                        yaw_rate: current_rate.2,
                        throttle: current_output.3,
                        roll_output: current_output.0,
                        pitch_output: current_output.1,
                        yaw_output: current_output.2,
                    });
                }
            }

            _ => {}
        }
    }

    Ok(data_points)
}

/// Helper function to get field value from CSV record
fn get_field_value(record: &csv::StringRecord, headers: &csv::StringRecord, field_names: &[&str]) -> Option<f64> {
    for name in field_names {
        if let Some(pos) = headers.iter().position(|h| h.eq_ignore_ascii_case(name)) {
            if let Some(value) = record.get(pos) {
                if let Ok(num) = value.parse::<f64>() {
                    return Some(num);
                }
            }
        }
    }
    None
}

/// Export flight data to CSV
pub fn export_to_csv(data: &[FlightDataPoint], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    // Write headers
    writer.write_record(&[
        "timestamp",
        "roll_desired", "roll_actual", "roll_error",
        "pitch_desired", "pitch_actual", "pitch_error",
        "yaw_desired", "yaw_actual", "yaw_error",
        "alt_desired", "alt_actual", "alt_error",
        "roll_rate", "pitch_rate", "yaw_rate",
        "throttle", "roll_output", "pitch_output", "yaw_output",
    ])?;

    // Write data
    for point in data {
        writer.write_record(&[
            point.timestamp.to_string(),
            point.roll_desired.to_string(),
            point.roll_actual.to_string(),
            point.roll_error().to_string(),
            point.pitch_desired.to_string(),
            point.pitch_actual.to_string(),
            point.pitch_error().to_string(),
            point.yaw_desired.to_string(),
            point.yaw_actual.to_string(),
            point.yaw_error().to_string(),
            point.alt_desired.to_string(),
            point.alt_actual.to_string(),
            point.alt_error().to_string(),
            point.roll_rate.to_string(),
            point.pitch_rate.to_string(),
            point.yaw_rate.to_string(),
            point.throttle.to_string(),
            point.roll_output.to_string(),
            point.pitch_output.to_string(),
            point.yaw_output.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}
