#!/usr/bin/env python3
"""
Final test suite to verify all functionality of the HMM futures program.
"""

import subprocess
import sys
from pathlib import Path

def run_test(name, cmd):
    """Run a test and report the result."""
    print(f"Running {name}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
    if result.returncode == 0:
        print(f"  ✓ {name} passed")
        return True
    else:
        print(f"  ✗ {name} failed")
        print(f"    Error: {result.stderr}")
        return False

def main():
    """Run all tests."""
    print("Running comprehensive test suite for HMM futures program...\\n")
    
    # Activate virtual environment
    env_setup = "source .venv/bin/activate && "
    
    tests = [
        ("Unit tests", f"{env_setup}python test_main.py"),
        ("CLI tests", f"{env_setup}python test_cli.py"),
        ("Lookahead bias prevention test", f"{env_setup}python test_lookahead.py"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, cmd in tests:
        if run_test(name, cmd):
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The HMM futures program is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())