#!/usr/bin/env python3

# Copyright (c) 2025 sylcrala | sylcrala.xyz
# Licensed under the MIT License - see LICENSE.md for details

"""
eBay CrossLister - Installation Script
Cross-platform setup for TUI application
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

VERSION = "1.0.0"
APP_NAME = "eBay CrossLister"

def print_header():
    print(" " + APP_NAME + f" v{VERSION}")
    print("=" * 50)
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f" Python: {sys.version.split()[0]}")
    print()

def check_python_version():
    """Ensure Python 3.8+ is available"""
    if sys.version_info < (3, 8):
        print(" Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python: https://www.python.org/downloads/")
        sys.exit(1)
    print(" Python version check passed")

def setup_virtual_environment(force_reinstall=False):
    """Create and setup virtual environment"""
    venv_path = Path(".venv")
    
    if force_reinstall and venv_path.exists():
        print("  Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    if not venv_path.exists():
        print(" Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print(" Virtual environment created")
    else:
        print(" Virtual environment already exists")
    
    return venv_path

def get_pip_executable(venv_path):
    """Get the correct pip executable path for the platform"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def get_python_executable(venv_path):
    """Get the correct python executable path for the platform"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def install_dependencies(venv_path, dev_mode=False):
    """Install required dependencies"""
    pip_exe = get_pip_executable(venv_path)
    python_exe = get_python_executable(venv_path)
    
    print(" Installing dependencies...")
    
    # Upgrade pip first
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install main requirements
    if Path("requirements.txt").exists():
        print("    Installing from requirements.txt...")
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
    else:
        print(" requirements.txt not found!")
        sys.exit(1)
    
    # Install development dependencies if requested
    if dev_mode:
        print("     Installing development tools...")
        dev_packages = ["black>=23.0.0", "pytest>=7.0.0", "mypy>=1.0.0"]
        subprocess.run([str(pip_exe), "install"] + dev_packages, check=True)
    
    print(" Dependencies installed successfully")

def create_launch_scripts(venv_path):
    """Create convenient launch scripts"""
    python_exe = get_python_executable(venv_path)
    
    # Cross-platform launch script
    if platform.system() == "Windows":
        # Windows batch script
        with open("run_crosslister.bat", "w") as f:
            f.write(f'@echo off\n')
            f.write(f'"{python_exe}" main.py %*\n')
        print("✅ Created run_crosslister.bat (Windows launcher)")
    else:
        # Unix shell script
        with open("run_crosslister.sh", "w") as f:
            f.write(f'#!/bin/bash\n')
            f.write(f'"{python_exe}" main.py "$@"\n')
        os.chmod("run_crosslister.sh", 0o755)
        print("✅ Created run_crosslister.sh (Unix launcher)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="eBay CrossLister Installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py              # Standard installation
  python install.py --dev        # Install with development tools
  python install.py --force      # Force reinstall (remove existing venv)
  
After installation:
  python main.py                 # Run the application
  ./run_crosslister.sh           # Unix shortcut (created during install)
  run_crosslister.bat            # Windows shortcut (created during install)
        """
    )
    
    parser.add_argument(
        "--dev", "-d",
        action="store_true",
        help="Install development dependencies (black, pytest, mypy)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true", 
        help="Force reinstall (remove existing virtual environment)"
    )
    
    args = parser.parse_args()
    
    try:
        print_header()
        
        print(" Checking system requirements...")
        check_python_version()
        
        print("\n Setting up environment...")
        venv_path = setup_virtual_environment(args.force)
        
        print("\n  Installing packages...")
        install_dependencies(venv_path, args.dev)
        
        print("\n Creating launch scripts...")
        create_launch_scripts(venv_path)
        
        print("\n" + "="*50)
        print(" Installation Complete!")
        print("="*50)
        
        python_exe = get_python_executable(venv_path)
        
        print(f"\n  To run {APP_NAME}:")
        print(f"   {python_exe} main.py")
        
        if platform.system() == "Windows":
            print("   run_crosslister.bat")
        else:
            print("   ./run_crosslister.sh")
        
        print(f"\n Development commands:")
        print(f"   {python_exe} main.py --help     # Show help")
        print(f"   {python_exe} main.py --config   # Open config")
        print(f"   {python_exe} main.py --setup    # Run setup wizard")
        print(f"   {python_exe} main.py --reset    # Reset configuration")
        
        if args.dev:
            print(f"\n  Development tools installed:")
            print(f"   {python_exe} -m black main.py    # Format code")
            print(f"   {python_exe} -m pytest           # Run tests")
            print(f"   {python_exe} -m mypy main.py     # Type checking")
        
        print(f"\n Documentation:")
        print("   README.md - Setup and usage guide")
        print("   docs/CONFIG_GUIDE.md - API configuration")
        
        print(f"\n Security:")
        print("   Credentials stored securely in OS keyring")
        print("   No sensitive data in configuration files")
        
    except subprocess.CalledProcessError as e:
        print(f"\n Installation failed: {e}")
        print("   Please check the error messages above and try again")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()