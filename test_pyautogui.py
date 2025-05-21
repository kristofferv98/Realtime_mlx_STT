#!/usr/bin/env python3
import time
import sys

print("Attempting to import PyAutoGUI...")
try:
    import pyautogui
    print(f"PyAutoGUI version: {pyautogui.__version__}")
    print("PyAutoGUI successfully imported")
    
    # Wait a moment before simulating keypress
    print("Waiting 3 seconds before simulating Command+V...")
    time.sleep(3)
    
    print("Simulating Command+V keypress")
    pyautogui.hotkey('command', 'v')
    print("Test completed successfully")
    
except ImportError as e:
    print(f"PyAutoGUI import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"PyAutoGUI encountered an error: {e}")
    sys.exit(2)