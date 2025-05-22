"""
Concrete implementation of the IEventBus interface.

This module provides the EventBus class, which is a thread-safe implementation
of the publish-subscribe pattern using the IEventBus interface.
"""

import logging
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Set, Type

from .event import Event
from ..Common.Interfaces.event_bus import IEventBus

# Configure logging
logger = logging.getLogger(__name__)


class EventBus(IEventBus):
    """
    Thread-safe implementation of the event bus.
    
    The EventBus allows components to publish events and subscribe to event types
    without being directly coupled to each other. It maintains a registry of
    event types and their subscribers.
    """
    
    def __init__(self):
        """Initialize the event bus with an empty subscription registry."""
        # Use defaultdict to automatically create an empty set for new event types
        self._subscriptions: Dict[Type[Event], Set[Callable[[Event], None]]] = defaultdict(set)
        self._lock = threading.RLock()  # Use RLock for reentrant locking
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        This method notifies all handlers that have subscribed to the event's type.
        If an exception occurs in a handler, it is logged but doesn't prevent other
        handlers from being notified.
        
        Args:
            event: The event to publish
        """
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event type, got {type(event).__name__}")
        
        # Get the event's class and all its parent classes that are Event subclasses
        event_classes = [cls for cls in type(event).__mro__ if cls is Event or (cls is not object and issubclass(cls, Event))]
        
        # Notify handlers for the event type and all parent event types
        for event_class in event_classes:
            self._notify_handlers(event, event_class)
    
    def _notify_handlers(self, event: Event, event_class: Type[Event]) -> None:
        """
        Notify all handlers subscribed to a specific event class.
        
        Args:
            event: The event instance to pass to handlers
            event_class: The event class to find handlers for
        """
        handlers = self._get_handlers_for_event(event_class)
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_class.__name__}: {str(e)}", exc_info=True)
    
    def _get_handlers_for_event(self, event_type: Type[Event]) -> List[Callable[[Event], None]]:
        """
        Get all handlers for a specific event type.
        
        Args:
            event_type: The event type to get handlers for
            
        Returns:
            List of handler functions for the event type
        """
        with self._lock:
            # Create a copy of the handlers to avoid modification during iteration
            return list(self._subscriptions.get(event_type, set()))
    
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: The callback function to invoke when the event occurs
        """
        if not issubclass(event_type, Event):
            raise TypeError(f"Expected Event subclass, got {event_type.__name__}")
        
        with self._lock:
            self._subscriptions[event_type].add(handler)
            logger.debug(f"Subscribed handler to {event_type.__name__}, total handlers: {len(self._subscriptions[event_type])}")
    
    def unsubscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> bool:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
            
        Returns:
            bool: True if the handler was successfully unsubscribed
        """
        if not issubclass(event_type, Event):
            raise TypeError(f"Expected Event subclass, got {event_type.__name__}")
        
        with self._lock:
            if event_type in self._subscriptions and handler in self._subscriptions[event_type]:
                self._subscriptions[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.__name__}, remaining handlers: {len(self._subscriptions[event_type])}")
                return True
            return False
    
    def clear_subscriptions(self) -> None:
        """Clear all event subscriptions."""
        with self._lock:
            self._subscriptions.clear()
            logger.debug("All event subscriptions cleared")
    
    def clear_subscriptions_for_type(self, event_type: Type[Event]) -> None:
        """
        Clear all subscriptions for a specific event type.
        
        Args:
            event_type: The event type to clear subscriptions for
        """
        if not issubclass(event_type, Event):
            raise TypeError(f"Expected Event subclass, got {event_type.__name__}")
        
        with self._lock:
            if event_type in self._subscriptions:
                self._subscriptions[event_type].clear()
                logger.debug(f"Cleared all subscriptions for {event_type.__name__}")
    
    def get_subscription_count(self, event_type: Type[Event] = None) -> int:
        """
        Get the count of subscriptions, optionally for a specific event type.
        
        Args:
            event_type: Optional specific event type to count subscriptions for
            
        Returns:
            int: Count of subscriptions
        """
        with self._lock:
            if event_type is not None:
                if not issubclass(event_type, Event):
                    raise TypeError(f"Expected Event subclass, got {event_type.__name__}")
                return len(self._subscriptions.get(event_type, set()))
            
            return sum(len(handlers) for handlers in self._subscriptions.values())