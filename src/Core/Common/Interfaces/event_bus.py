"""
Event Bus interface for the event-driven architecture.

This module defines the IEventBus interface that abstracts the publish-subscribe 
pattern for event handling in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, ForwardRef

# Use forward reference for Event
Event = ForwardRef('Event')



class IEventBus(ABC):
    """
    Interface for the event bus that enables communication between features.
    
    The event bus implements the publish-subscribe pattern, allowing components
    to publish events and subscribe to event types they're interested in.
    """
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        pass
    
    @abstractmethod
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: The callback function to invoke when the event occurs
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> bool:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
            
        Returns:
            bool: True if the handler was successfully unsubscribed
        """
        pass
    
    @abstractmethod
    def clear_subscriptions(self) -> None:
        """
        Clear all event subscriptions.
        """
        pass