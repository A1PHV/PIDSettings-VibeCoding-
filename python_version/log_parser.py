"""
Ardupilot log parser
Supports .bin (DataFlash), .log (text), and .csv formats
"""

import csv
from pathlib import Path
from typing import List
from models import FlightDataPoint


def parse_log(path: Path) -> List[FlightDataPoint]:
    """Parse Ardupilot log file - auto-detects format"""
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == '.bin':
        return parse_bin_log(path)
    elif suffix == '.csv':
        return parse_csv_log(path)
    elif suffix in ['.log', '.txt']:
        return parse_text_log(path)
    else:
        # Try CSV first, then text
        try:
            return parse_csv_log(path)
        except:
            return parse_text_log(path)


def parse_bin_log(path: Path) -> List[FlightDataPoint]:
    """Parse binary DataFlash log (.bin format)"""
    try:
        from pymavlink import mavutil
    except ImportError:
        raise ImportError(
            "pymavlink is required to parse .bin files. "
            "Install it with: pip install pymavlink"
        )

    print(f"   Reading binary log with pymavlink...")
    mlog = mavutil.mavlink_connection(str(path))

    data_points = []

    # Temporary storage for different message types
    current_data = {
        'timestamp': 0.0,
        'roll_desired': 0.0, 'roll_actual': 0.0,
        'pitch_desired': 0.0, 'pitch_actual': 0.0,
        'yaw_desired': 0.0, 'yaw_actual': 0.0,
        'alt_desired': 0.0, 'alt_actual': 0.0,
        'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0,
        'throttle': 0.0,
        'roll_output': 0.0, 'pitch_output': 0.0, 'yaw_output': 0.0,
    }

    message_count = 0
    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break

        message_count += 1
        if message_count % 10000 == 0:
            print(f"   Processed {message_count} messages, {len(data_points)} data points...")

        msg_type = msg.get_type()

        # ATT - Attitude
        if msg_type == 'ATT':
            current_data['timestamp'] = getattr(msg, 'TimeUS', 0) / 1_000_000.0
            current_data['roll_desired'] = getattr(msg, 'DesRoll', 0.0)
            current_data['roll_actual'] = getattr(msg, 'Roll', 0.0)
            current_data['pitch_desired'] = getattr(msg, 'DesPitch', 0.0)
            current_data['pitch_actual'] = getattr(msg, 'Pitch', 0.0)
            current_data['yaw_desired'] = getattr(msg, 'DesYaw', 0.0)
            current_data['yaw_actual'] = getattr(msg, 'Yaw', 0.0)

        # RATE - Angular rates
        elif msg_type == 'RATE':
            current_data['roll_rate'] = getattr(msg, 'R', 0.0)
            current_data['pitch_rate'] = getattr(msg, 'P', 0.0)
            current_data['yaw_rate'] = getattr(msg, 'Y', 0.0)

        # CTUN - Control tuning (altitude)
        elif msg_type == 'CTUN':
            current_data['alt_desired'] = getattr(msg, 'DAlt', 0.0)
            current_data['alt_actual'] = getattr(msg, 'Alt', 0.0)
            current_data['throttle'] = getattr(msg, 'ThO', 0.0) / 1000.0  # Normalize

        # RCOU - RC Output
        elif msg_type == 'RCOU':
            # Channels typically: 1=Roll, 2=Pitch, 3=Throttle, 4=Yaw
            current_data['roll_output'] = getattr(msg, 'C1', 0.0)
            current_data['pitch_output'] = getattr(msg, 'C2', 0.0)
            current_data['yaw_output'] = getattr(msg, 'C4', 0.0)

            # Create data point when we have output data
            if current_data['timestamp'] > 0:
                data_points.append(FlightDataPoint(**current_data))

    print(f"   Finished processing {message_count} messages")
    return data_points


def parse_csv_log(path: Path) -> List[FlightDataPoint]:
    """Parse CSV format log (exported from Mission Planner)"""
    data_points = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)

        # Get all headers (case-insensitive lookup)
        headers = {h.lower(): h for h in reader.fieldnames}

        for row in reader:
            # Helper to get field value
            def get_field(names, default=0.0):
                for name in names:
                    if name.lower() in headers:
                        actual_name = headers[name.lower()]
                        try:
                            return float(row[actual_name])
                        except (ValueError, KeyError):
                            pass
                return default

            timestamp = get_field(['timestamp', 'time', 'timeus']) / 1_000_000.0

            point = FlightDataPoint(
                timestamp=timestamp,
                roll_desired=get_field(['desroll', 'rollin', 'roll_des']),
                roll_actual=get_field(['roll']),
                pitch_desired=get_field(['despitch', 'pitchin', 'pitch_des']),
                pitch_actual=get_field(['pitch']),
                yaw_desired=get_field(['desyaw', 'yawin', 'yaw_des']),
                yaw_actual=get_field(['yaw']),
                alt_desired=get_field(['dalt', 'alt_des', 'posd']),
                alt_actual=get_field(['alt', 'altitude']),
                roll_rate=get_field(['gyrx', 'rollrate', 'gyro_x']),
                pitch_rate=get_field(['gyry', 'pitchrate', 'gyro_y']),
                yaw_rate=get_field(['gyrz', 'yawrate', 'gyro_z']),
                throttle=get_field(['tho', 'throut', 'throttle']),
                roll_output=get_field(['rout', 'rollout', 'c1']),
                pitch_output=get_field(['pout', 'pitchout', 'c2']),
                yaw_output=get_field(['yout', 'yawout', 'c4']),
            )

            data_points.append(point)

    return data_points


def parse_text_log(path: Path) -> List[FlightDataPoint]:
    """Parse text format log (native .log format)"""
    data_points = []

    # Temporary storage
    current_att = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                   'des_roll': 0.0, 'des_pitch': 0.0, 'des_yaw': 0.0}
    current_rate = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    current_alt = {'desired': 0.0, 'actual': 0.0}
    current_output = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'throttle': 0.0}
    timestamp = 0.0

    with open(path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if not parts:
                continue

            msg_type = parts[0]

            try:
                if msg_type == 'ATT' and len(parts) > 7:
                    # ATT, TimeUS, DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw
                    timestamp = float(parts[1]) / 1_000_000.0
                    current_att['des_roll'] = float(parts[2])
                    current_att['roll'] = float(parts[3])
                    current_att['des_pitch'] = float(parts[4])
                    current_att['pitch'] = float(parts[5])
                    current_att['des_yaw'] = float(parts[6])
                    current_att['yaw'] = float(parts[7])

                elif msg_type == 'RATE' and len(parts) > 7:
                    # RATE, TimeUS, RDes, R, PDes, P, YDes, Y
                    current_rate['roll'] = float(parts[3])
                    current_rate['pitch'] = float(parts[5])
                    current_rate['yaw'] = float(parts[7])

                elif msg_type == 'CTUN' and len(parts) > 3:
                    # CTUN, TimeUS, ThI, ThO, DAlt, Alt, ...
                    current_alt['desired'] = float(parts[4]) if len(parts) > 4 else 0.0
                    current_alt['actual'] = float(parts[5]) if len(parts) > 5 else 0.0
                    if len(parts) > 3:
                        current_output['throttle'] = float(parts[3]) / 1000.0

                elif msg_type == 'RCOU' and len(parts) > 4:
                    # RCOU, TimeUS, C1, C2, C3, C4, ...
                    current_output['roll'] = float(parts[2])
                    current_output['pitch'] = float(parts[3])
                    current_output['yaw'] = float(parts[5]) if len(parts) > 5 else 0.0

                    # Create data point
                    if timestamp > 0:
                        point = FlightDataPoint(
                            timestamp=timestamp,
                            roll_desired=current_att['des_roll'],
                            roll_actual=current_att['roll'],
                            pitch_desired=current_att['des_pitch'],
                            pitch_actual=current_att['pitch'],
                            yaw_desired=current_att['des_yaw'],
                            yaw_actual=current_att['yaw'],
                            alt_desired=current_alt['desired'],
                            alt_actual=current_alt['actual'],
                            roll_rate=current_rate['roll'],
                            pitch_rate=current_rate['pitch'],
                            yaw_rate=current_rate['yaw'],
                            throttle=current_output['throttle'],
                            roll_output=current_output['roll'],
                            pitch_output=current_output['pitch'],
                            yaw_output=current_output['yaw'],
                        )
                        data_points.append(point)

            except (ValueError, IndexError):
                continue

    return data_points


def export_to_csv(data: List[FlightDataPoint], path: Path):
    """Export flight data to CSV"""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'timestamp',
            'roll_desired', 'roll_actual', 'roll_error',
            'pitch_desired', 'pitch_actual', 'pitch_error',
            'yaw_desired', 'yaw_actual', 'yaw_error',
            'alt_desired', 'alt_actual', 'alt_error',
            'roll_rate', 'pitch_rate', 'yaw_rate',
            'throttle', 'roll_output', 'pitch_output', 'yaw_output',
        ])

        # Write data
        for point in data:
            writer.writerow([
                point.timestamp,
                point.roll_desired, point.roll_actual, point.roll_error(),
                point.pitch_desired, point.pitch_actual, point.pitch_error(),
                point.yaw_desired, point.yaw_actual, point.yaw_error(),
                point.alt_desired, point.alt_actual, point.alt_error(),
                point.roll_rate, point.pitch_rate, point.yaw_rate,
                point.throttle, point.roll_output, point.pitch_output, point.yaw_output,
            ])
