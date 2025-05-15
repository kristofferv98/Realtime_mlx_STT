from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeviceInfo:
    """Information about an audio input device."""
    
    # Device identifier
    id: int
    
    # Device name
    name: str
    
    # Maximum input channels
    max_input_channels: int
    
    # Default sample rate
    default_sample_rate: int
    
    # Supported sample rates
    supported_sample_rates: List[int]
    
    # Is this the default device?
    is_default: bool = False
    
    # Additional device-specific information
    extra_info: Optional[dict] = None