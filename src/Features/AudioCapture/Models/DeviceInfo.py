from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class DeviceInfo:
    """
    Represents information about an audio input device.
    
    This class encapsulates all relevant information about an audio capture device,
    providing a standardized interface for device selection and configuration.
    """
    
    # Unique identifier for the device
    device_id: int
    
    # Human-readable name of the device
    name: str
    
    # Maximum number of input channels supported
    max_input_channels: int
    
    # Default sample rate for this device
    default_sample_rate: int
    
    # List of supported sample rates
    supported_sample_rates: List[int]
    
    # Whether this is the system default input device
    is_default: bool = False
    
    # Optional additional device information (driver-specific)
    additional_info: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """
        Create a string representation of the device suitable for user presentation.
        
        Returns:
            str: Formatted string with device information
        """
        rates_str = ", ".join(str(rate) for rate in self.supported_sample_rates)
        return (
            f"Device {self.device_id}: {self.name} "
            f"(Channels: {self.max_input_channels}, "
            f"Default rate: {self.default_sample_rate}Hz, "
            f"Supported rates: {rates_str})"
            f"{' [DEFAULT]' if self.is_default else ''}"
        )
    
    @classmethod
    def from_pyaudio_device_info(cls, device_id: int, device_info: Dict[str, Any], 
                                supported_rates: List[int], is_default: bool = False) -> 'DeviceInfo':
        """
        Create a DeviceInfo instance from PyAudio device information.
        
        Args:
            device_id: The device index
            device_info: PyAudio device info dictionary
            supported_rates: List of supported sample rates
            is_default: Whether this is the default device
            
        Returns:
            DeviceInfo: A new device info instance
        """
        return cls(
            device_id=device_id,
            name=device_info.get('name', 'Unknown Device'),
            max_input_channels=device_info.get('maxInputChannels', 0),
            default_sample_rate=int(device_info.get('defaultSampleRate', 44100)),
            supported_sample_rates=supported_rates,
            is_default=is_default,
            additional_info={
                k: v for k, v in device_info.items() 
                if k not in ('name', 'maxInputChannels', 'defaultSampleRate')
            }
        )