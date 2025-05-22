#!/usr/bin/env python3
"""
Test script to verify VAD configuration flow from UI to detectors.

This script tests that:
1. Lazy initialization works correctly
2. Configuration parameters are properly passed through
3. All new parameters are configurable
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Features.VoiceActivityDetection import VadModule
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Infrastructure.Logging import LoggingModule

def test_vad_configuration():
    """Test the complete VAD configuration flow."""
    print("Testing VAD Configuration Flow")
    print("=" * 50)
    
    # Initialize components
    event_bus = EventBus()
    command_dispatcher = CommandDispatcher()
    
    # Register VAD module
    handler = VadModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        default_detector="combined",
        default_sensitivity=0.7,
        processing_enabled=False
    )
    
    print("\n1. Testing lazy initialization...")
    # At this point, detectors should not be created yet
    assert len(handler.detectors) == 0, "Detectors should not be initialized yet"
    print("✓ Detectors are not initialized on startup")
    
    print("\n2. Testing configuration with all parameters...")
    # Configure with all new parameters
    config_command = ConfigureVadCommand(
        detector_type="combined",
        sensitivity=0.8,
        min_speech_duration=0.3,
        window_size=10,
        parameters={
            "webrtc_aggressiveness": 1,
            "silero_threshold": 0.5,
            "webrtc_threshold": 0.7,
            "frame_duration_ms": 20,
            "speech_confirmation_frames": 3,
            "silence_confirmation_frames": 40,
            "speech_buffer_size": 150
        }
    )
    
    result = command_dispatcher.dispatch(config_command)
    assert result, "Configuration should succeed"
    print("✓ Configuration command accepted")
    
    print("\n3. Testing detector creation on first use...")
    # Get the detector - this should trigger lazy creation
    detector = handler._get_detector("combined")
    assert detector is not None, "Detector should be created"
    assert "combined" in handler.detectors, "Detector should be in registry"
    print("✓ Detector created lazily on first access")
    
    print("\n4. Verifying configuration was applied...")
    # Check if configuration was stored
    config = handler.detector_configs["combined"]
    assert config["webrtc_aggressiveness"] == 1
    assert config["silero_threshold"] == 0.5
    assert config["webrtc_threshold"] == 0.7
    assert config["frame_duration_ms"] == 20
    assert config["speech_confirmation_frames"] == 3
    assert config["silence_confirmation_frames"] == 40
    assert config["speech_buffer_size"] == 150
    print("✓ All configuration parameters stored correctly")
    
    print("\n5. Testing reconfiguration of existing detector...")
    # Reconfigure with different parameters
    reconfig_command = ConfigureVadCommand(
        detector_type="combined",
        sensitivity=0.5,
        parameters={
            "frame_duration_ms": 40,
            "speech_confirmation_frames": 1
        }
    )
    
    result = command_dispatcher.dispatch(reconfig_command)
    assert result, "Reconfiguration should succeed"
    
    # Verify new config
    config = handler.detector_configs["combined"]
    assert config["frame_duration_ms"] == 40
    assert config["speech_confirmation_frames"] == 1
    print("✓ Detector reconfiguration works")
    
    print("\n6. Testing other detector types...")
    # Configure WebRTC detector
    webrtc_command = ConfigureVadCommand(
        detector_type="webrtc",
        sensitivity=0.6,
        parameters={
            "frame_duration_ms": 30
        }
    )
    
    result = command_dispatcher.dispatch(webrtc_command)
    assert result, "WebRTC configuration should succeed"
    
    # Configure Silero detector
    silero_command = ConfigureVadCommand(
        detector_type="silero",
        sensitivity=0.7,
        parameters={
            "min_speech_duration_ms": 300
        }
    )
    
    result = command_dispatcher.dispatch(silero_command)
    assert result, "Silero configuration should succeed"
    print("✓ All detector types can be configured")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("\nConfiguration flow is working correctly:")
    print("- Lazy initialization prevents unnecessary resource usage")
    print("- All parameters can be configured from the application layer")
    print("- Detectors can be reconfigured after creation")
    print("- Frame processing parameters are now fully configurable")

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        test_vad_configuration()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)