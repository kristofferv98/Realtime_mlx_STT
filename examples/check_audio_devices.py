#!/usr/bin/env python3
"""
Script to check audio devices and their structure.
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature import
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule

def main():
    """List audio devices and show their structure."""
    # Initialize components
    command_dispatcher = CommandDispatcher()
    event_bus = EventBus()
    
    # Register audio capture module
    AudioCaptureModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus
    )
    
    # Get devices
    print("Getting devices from AudioCaptureModule...")
    devices = AudioCaptureModule.list_devices(command_dispatcher)
    
    # Show device count
    print(f"\nFound {len(devices)} devices")
    print(f"Return type: {type(devices)}")
    
    # Show detailed device info
    print("\nDetailed device information:")
    for i, device in enumerate(devices):
        print(f"\nDevice {i}:")
        print(f"  Type: {type(device)}")
        
        # Handle different return types
        if isinstance(device, dict):
            print(f"  Keys: {list(device.keys())}")
            # Print all device info
            for key, value in device.items():
                print(f"  {key}: {value}")
        elif isinstance(device, list):
            print(f"  List content: {device}")
        else:
            print(f"  Value: {device}")
    
    # Now check using PyAudio directly
    print("\n\nDirect PyAudio devices:")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                print(f"\nDevice {i}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
    except ImportError:
        print("PyAudio not available")
    except Exception as e:
        print(f"Error accessing PyAudio: {e}")

if __name__ == "__main__":
    main()