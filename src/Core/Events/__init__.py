"""
Events module for the Realtime_mlx_STT project.

This module provides event-related functionality for the event-driven architecture,
including the base Event class and EventBus implementation.
"""

from .event import Event
from .event_bus import EventBus

__all__ = [
    'Event',
    'EventBus'
]