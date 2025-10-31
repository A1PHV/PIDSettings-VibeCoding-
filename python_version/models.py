"""
Data models for PID Tuner
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path


@dataclass
class FlightDataPoint:
    """Single flight data point from Ardupilot log"""
    timestamp: float

    # Attitude (desired vs actual)
    roll_desired: float = 0.0
    roll_actual: float = 0.0
    pitch_desired: float = 0.0
    pitch_actual: float = 0.0
    yaw_desired: float = 0.0
    yaw_actual: float = 0.0

    # Altitude
    alt_desired: float = 0.0
    alt_actual: float = 0.0

    # Rate (gyro)
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    # Control outputs
    throttle: float = 0.0
    roll_output: float = 0.0
    pitch_output: float = 0.0
    yaw_output: float = 0.0

    def roll_error(self) -> float:
        return self.roll_desired - self.roll_actual

    def pitch_error(self) -> float:
        return self.pitch_desired - self.pitch_actual

    def yaw_error(self) -> float:
        return self.yaw_desired - self.yaw_actual

    def alt_error(self) -> float:
        return self.alt_desired - self.alt_actual


@dataclass
class PIDCoefficients:
    """PID coefficients for a single axis"""
    p: float
    i: float
    d: float

    def __str__(self):
        return f"P: {self.p:.4f}, I: {self.i:.4f}, D: {self.d:.4f}"


@dataclass
class PIDResults:
    """Complete PID tuning results for all axes"""
    roll: Optional[PIDCoefficients] = None
    pitch: Optional[PIDCoefficients] = None
    yaw: Optional[PIDCoefficients] = None
    altitude: Optional[PIDCoefficients] = None

    method: str = ""
    analysis_notes: List[str] = None

    def __post_init__(self):
        if self.analysis_notes is None:
            self.analysis_notes = []

    def __str__(self):
        lines = []
        lines.append(f"Method: {self.method}")
        lines.append("─" * 40)

        if self.roll:
            lines.append(f"Roll:     {self.roll}")
        if self.pitch:
            lines.append(f"Pitch:    {self.pitch}")
        if self.yaw:
            lines.append(f"Yaw:      {self.yaw}")
        if self.altitude:
            lines.append(f"Altitude: {self.altitude}")

        if self.analysis_notes:
            lines.append("\n📝 Notes:")
            for note in self.analysis_notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)

    def save_to_file(self, path: Path):
        """Save results to JSON file"""
        data = {
            'method': self.method,
            'roll': asdict(self.roll) if self.roll else None,
            'pitch': asdict(self.pitch) if self.pitch else None,
            'yaw': asdict(self.yaw) if self.yaw else None,
            'altitude': asdict(self.altitude) if self.altitude else None,
            'analysis_notes': self.analysis_notes
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def to_ardupilot_params(self) -> str:
        """Convert to Ardupilot parameter format"""
        params = []

        if self.roll:
            params.append(f"ATC_RAT_RLL_P {self.roll.p:.4f}")
            params.append(f"ATC_RAT_RLL_I {self.roll.i:.4f}")
            params.append(f"ATC_RAT_RLL_D {self.roll.d:.4f}")

        if self.pitch:
            params.append(f"ATC_RAT_PIT_P {self.pitch.p:.4f}")
            params.append(f"ATC_RAT_PIT_I {self.pitch.i:.4f}")
            params.append(f"ATC_RAT_PIT_D {self.pitch.d:.4f}")

        if self.yaw:
            params.append(f"ATC_RAT_YAW_P {self.yaw.p:.4f}")
            params.append(f"ATC_RAT_YAW_I {self.yaw.i:.4f}")
            params.append(f"ATC_RAT_YAW_D {self.yaw.d:.4f}")

        if self.altitude:
            params.append(f"PSC_POSZ_P {self.altitude.p:.4f}")
            params.append(f"PSC_VELZ_P {self.altitude.i:.4f}")
            params.append(f"PSC_ACCZ_P {self.altitude.d:.4f}")

        return "\n".join(params) + "\n"
