#!/usr/bin/env python3
"""
PID Tuner for Ardupilot - Python Version
Automatically calculates optimal PID coefficients from flight logs
"""

import argparse
import sys
import json
from pathlib import Path

# Import our modules
from log_parser import parse_log, export_to_csv
from pid_calculator import calculate_pid_coefficients
from models import PIDResults


def main():
    parser = argparse.ArgumentParser(
        prog='pid_tuner',
        description='Ardupilot PID coefficient calculator from flight logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Advanced Tuning Methods:
  auto            - Automatically select best method (RECOMMENDED)
  imc             - Internal Model Control (robust, good for delays)
  simc            - Skogestad IMC (disturbance rejection)
  cohen-coon      - Cohen-Coon (large dead time systems)
  lambda          - Lambda tuning (explicit speed control)
  tyreus-luyben   - Tyreus-Luyben (conservative, stable)
  optimization    - Numerical optimization (best performance)
  frequency       - Frequency domain (stability margins)
  ziegler-nichols - Classic Ziegler-Nichols
  relay           - Relay feedback method

Examples:
  %(prog)s analyze -i flight.bin
  %(prog)s analyze -i flight.bin -m auto
  %(prog)s analyze -i flight.log -a roll -m imc
  %(prog)s extract -i flight.bin -o data.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze log and calculate PID coefficients')
    analyze_parser.add_argument('-i', '--input', required=True, type=Path,
                                help='Path to Ardupilot log file (.bin, .log, or .csv)')
    analyze_parser.add_argument('-o', '--output', type=Path,
                                help='Output file for calculated PID coefficients (JSON)')
    analyze_parser.add_argument('-a', '--axis', default='all',
                                choices=['all', 'roll', 'pitch', 'yaw', 'alt'],
                                help='Axis to tune (default: all)')
    analyze_parser.add_argument('-m', '--method', default='auto',
                                choices=['auto', 'imc', 'simc', 'cohen-coon', 'lambda',
                                        'tyreus-luyben', 'optimization', 'frequency',
                                        'ziegler-nichols', 'relay'],
                                help='Tuning method (default: auto - automatically selects best)')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract flight data to CSV')
    extract_parser.add_argument('-i', '--input', required=True, type=Path,
                                help='Path to Ardupilot log file')
    extract_parser.add_argument('-o', '--output', required=True, type=Path,
                                help='Output CSV file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'analyze':
            analyze_command(args)
        elif args.command == 'extract':
            extract_command(args)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def analyze_command(args):
    """Analyze log and calculate PID coefficients"""
    print(f"🚁 Analyzing Ardupilot log: {args.input}")
    print(f"📊 Axis: {args.axis}")
    print(f"🔧 Method: {args.method}")

    # Parse log file
    print("\n⏳ Parsing log file...")
    flight_data = parse_log(args.input)
    print(f"✅ Parsed {len(flight_data)} data points")

    if len(flight_data) < 10:
        raise ValueError("Insufficient data points. Need at least 10 data points with active flight.")

    # Calculate PID coefficients
    print(f"\n⏳ Calculating PID coefficients using {args.method} method...")
    pid_results = calculate_pid_coefficients(flight_data, args.axis, args.method)

    # Print results
    print("\n" + "="*50)
    print("📈 Calculated PID Coefficients:")
    print("="*50)
    print(pid_results)

    # Save to file if specified
    if args.output:
        pid_results.save_to_file(args.output)
        print(f"\n💾 Results saved to: {args.output}")

        # Also save Ardupilot params format
        params_file = args.output.with_suffix('.params')
        with open(params_file, 'w') as f:
            f.write(pid_results.to_ardupilot_params())
        print(f"💾 Ardupilot parameters saved to: {params_file}")

    print("\n" + "="*50)
    print("⚠️  ВАЖНО: Начните с 70% от рассчитанных значений!")
    print("="*50)


def extract_command(args):
    """Extract flight data to CSV"""
    print(f"📝 Extracting data from: {args.input}")

    flight_data = parse_log(args.input)
    export_to_csv(flight_data, args.output)

    print(f"✅ Data exported to: {args.output}")
    print(f"   Total points: {len(flight_data)}")


if __name__ == '__main__':
    main()
