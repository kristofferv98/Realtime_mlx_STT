#!/usr/bin/env python3
"""
Utility script for changing log levels at runtime.

This script can be used to change log levels for running applications
that use the centralized logging system.

Usage:
    python change_log_level.py [feature_name] [level]
    
Examples:
    python change_log_level.py AudioCapture DEBUG
    python change_log_level.py Transcription INFO
    python change_log_level.py root WARNING
"""

import os
import sys
import argparse
import socket
import json
import time

# Default port for log control
LOG_CONTROL_PORT = 50101


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Change log levels at runtime')
    parser.add_argument('target', help='Target feature or "root" for root logger')
    parser.add_argument('level', help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--port', type=int, default=LOG_CONTROL_PORT,
                      help=f'Port to send command to (default: {LOG_CONTROL_PORT})')
    return parser.parse_args()


def validate_level(level):
    """Validate the log level."""
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        print(f"Error: Invalid log level '{level}'. Must be one of {', '.join(valid_levels)}")
        return False
    return True


def send_command(target, level, port):
    """Send log level change command via UDP."""
    command = {
        'action': 'set_level',
        'target': target,
        'level': level.upper(),
        'timestamp': time.time()
    }
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        message = json.dumps(command).encode('utf-8')
        sock.sendto(message, ('127.0.0.1', port))
        print(f"Command sent: Set log level for {target} to {level.upper()}")
        return True
    except Exception as e:
        print(f"Error sending command: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    if not validate_level(args.level):
        return 1
    
    success = send_command(args.target, args.level, args.port)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())