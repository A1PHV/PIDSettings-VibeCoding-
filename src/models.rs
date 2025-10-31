use serde::{Deserialize, Serialize};
use std::fmt;

/// Flight data point from Ardupilot log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightDataPoint {
    pub timestamp: f64,  // Time in seconds

    // Attitude (desired vs actual)
    pub roll_desired: f64,
    pub roll_actual: f64,
    pub pitch_desired: f64,
    pub pitch_actual: f64,
    pub yaw_desired: f64,
    pub yaw_actual: f64,

    // Altitude
    pub alt_desired: f64,
    pub alt_actual: f64,

    // Rate (gyro)
    pub roll_rate: f64,
    pub pitch_rate: f64,
    pub yaw_rate: f64,

    // Control outputs
    pub throttle: f64,
    pub roll_output: f64,
    pub pitch_output: f64,
    pub yaw_output: f64,
}

impl FlightDataPoint {
    pub fn roll_error(&self) -> f64 {
        self.roll_desired - self.roll_actual
    }

    pub fn pitch_error(&self) -> f64 {
        self.pitch_desired - self.pitch_actual
    }

    pub fn yaw_error(&self) -> f64 {
        self.yaw_desired - self.yaw_actual
    }

    pub fn alt_error(&self) -> f64 {
        self.alt_desired - self.alt_actual
    }
}

/// PID coefficients for a single axis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDCoefficients {
    pub p: f64,  // Proportional
    pub i: f64,  // Integral
    pub d: f64,  // Derivative
}

impl fmt::Display for PIDCoefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P: {:.4}, I: {:.4}, D: {:.4}", self.p, self.i, self.d)
    }
}

/// Complete PID tuning results for all axes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDResults {
    pub roll: Option<PIDCoefficients>,
    pub pitch: Option<PIDCoefficients>,
    pub yaw: Option<PIDCoefficients>,
    pub altitude: Option<PIDCoefficients>,

    pub method: String,
    pub analysis_notes: Vec<String>,
}

impl fmt::Display for PIDResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Method: {}", self.method)?;
        writeln!(f, "─────────────────────────────────")?;

        if let Some(ref roll) = self.roll {
            writeln!(f, "Roll:    {}", roll)?;
        }
        if let Some(ref pitch) = self.pitch {
            writeln!(f, "Pitch:   {}", pitch)?;
        }
        if let Some(ref yaw) = self.yaw {
            writeln!(f, "Yaw:     {}", yaw)?;
        }
        if let Some(ref alt) = self.altitude {
            writeln!(f, "Altitude: {}", alt)?;
        }

        if !self.analysis_notes.is_empty() {
            writeln!(f, "\n📝 Notes:")?;
            for note in &self.analysis_notes {
                writeln!(f, "  • {}", note)?;
            }
        }

        Ok(())
    }
}

impl PIDResults {
    pub fn save_to_file(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn to_ardupilot_params(&self) -> String {
        let mut params = String::new();

        if let Some(ref roll) = self.roll {
            params.push_str(&format!("ATC_RAT_RLL_P {:.4}\n", roll.p));
            params.push_str(&format!("ATC_RAT_RLL_I {:.4}\n", roll.i));
            params.push_str(&format!("ATC_RAT_RLL_D {:.4}\n", roll.d));
        }

        if let Some(ref pitch) = self.pitch {
            params.push_str(&format!("ATC_RAT_PIT_P {:.4}\n", pitch.p));
            params.push_str(&format!("ATC_RAT_PIT_I {:.4}\n", pitch.i));
            params.push_str(&format!("ATC_RAT_PIT_D {:.4}\n", pitch.d));
        }

        if let Some(ref yaw) = self.yaw {
            params.push_str(&format!("ATC_RAT_YAW_P {:.4}\n", yaw.p));
            params.push_str(&format!("ATC_RAT_YAW_I {:.4}\n", yaw.i));
            params.push_str(&format!("ATC_RAT_YAW_D {:.4}\n", yaw.d));
        }

        if let Some(ref alt) = self.altitude {
            params.push_str(&format!("PSC_POSZ_P {:.4}\n", alt.p));
            params.push_str(&format!("PSC_VELZ_P {:.4}\n", alt.i));
            params.push_str(&format!("PSC_ACCZ_P {:.4}\n", alt.d));
        }

        params
    }
}
