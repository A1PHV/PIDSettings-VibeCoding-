#!/usr/bin/env python3
"""
Script to build standalone .exe using PyInstaller
Run this on a machine with internet to create portable .exe
"""

import sys
import subprocess
from pathlib import Path


def build_exe():
    """Build standalone executable"""

    print("=" * 60)
    print("Building PID Tuner executable")
    print("=" * 60)

    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"✅ PyInstaller found: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller not found!")
        print("\nInstalling PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller installed")

    # Build command
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single file
        "--console",                    # Console app
        "--name", "pid_tuner",          # Output name
        "--clean",                      # Clean cache
        # Hidden imports for pymavlink
        "--hidden-import", "pymavlink",
        "--hidden-import", "pymavlink.mavutil",
        "--hidden-import", "pymavlink.dialects",
        "--hidden-import", "pymavlink.dialects.v20",
        "--hidden-import", "pymavlink.dialects.v20.ardupilotmega",
        # Main script
        "pid_tuner.py"
    ]

    print(f"\n📦 Building executable...")
    print(f"   Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.check_call(cmd)

        print("\n" + "=" * 60)
        print("✅ Build successful!")
        print("=" * 60)

        # Find output
        exe_path = Path("dist") / "pid_tuner.exe"
        if not exe_path.exists():
            exe_path = Path("dist") / "pid_tuner"  # Linux/Mac

        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\n📦 Executable created: {exe_path}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"\n🚀 You can now copy '{exe_path.name}' to any computer!")
            print("   No Python or internet required!")
        else:
            print("\n⚠️  Executable not found in expected location")
            print("   Check the 'dist' folder")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(build_exe())
