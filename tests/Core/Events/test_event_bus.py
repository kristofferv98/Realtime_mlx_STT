#!/usr/bin/env python3
"""
Unit tests for EventBus.

This module tests the event bus implementation,
including publishing, subscribing, and thread safety.
"""

import unittest
from unittest.mock import Mock, MagicMock
import threading
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.Core.Events.event_bus import EventBus
from src.Core.Events.event import Event


# Test event classes
class TestEvent(Event):
    """Test event for unit tests."""
    def __init__(self, data: str):
        super().__init__(name="TestEvent")
        self.data = data


class AnotherTestEvent(Event):
    """Another test event for unit tests."""
    def __init__(self, value: int):
        super().__init__(name="AnotherTestEvent")
        self.value = value


class DerivedTestEvent(TestEvent):
    """Derived test event to test inheritance."""
    def __init__(self, data: str, extra: str):
        super().__init__(data)
        self.extra = extra


class TestEventBus(unittest.TestCase):
    """Test cases for EventBus."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        
    def tearDown(self):
        """Clean up after tests."""
        self.event_bus.clear_subscriptions()
        
    def test_initialization(self):
        """Test event bus initialization."""
        event_bus = EventBus()
        self.assertIsNotNone(event_bus)
        self.assertEqual(event_bus.get_subscription_count(), 0)
        
    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish functionality."""
        handler_called = False
        received_event = None
        
        def test_handler(event):
            nonlocal handler_called, received_event
            handler_called = True
            received_event = event
            
        # Subscribe to event
        self.event_bus.subscribe(TestEvent, test_handler)
        
        # Publish event
        test_event = TestEvent("test data")
        self.event_bus.publish(test_event)
        
        # Verify handler was called
        self.assertTrue(handler_called)
        self.assertEqual(received_event, test_event)
        self.assertEqual(received_event.data, "test data")
        
    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event type."""
        calls = []
        
        def handler1(event):
            calls.append(('handler1', event.data))
            
        def handler2(event):
            calls.append(('handler2', event.data))
            
        def handler3(event):
            calls.append(('handler3', event.data))
            
        # Subscribe multiple handlers
        self.event_bus.subscribe(TestEvent, handler1)
        self.event_bus.subscribe(TestEvent, handler2)
        self.event_bus.subscribe(TestEvent, handler3)
        
        # Publish event
        self.event_bus.publish(TestEvent("test"))
        
        # Verify all handlers were called
        self.assertEqual(len(calls), 3)
        self.assertIn(('handler1', 'test'), calls)
        self.assertIn(('handler2', 'test'), calls)
        self.assertIn(('handler3', 'test'), calls)
        
    def test_different_event_types(self):
        """Test different event types don't interfere."""
        test_calls = []
        another_calls = []
        
        def test_handler(event):
            test_calls.append(event.data)
            
        def another_handler(event):
            another_calls.append(event.value)
            
        # Subscribe to different event types
        self.event_bus.subscribe(TestEvent, test_handler)
        self.event_bus.subscribe(AnotherTestEvent, another_handler)
        
        # Publish different events
        self.event_bus.publish(TestEvent("test"))
        self.event_bus.publish(AnotherTestEvent(42))
        
        # Verify correct handlers were called
        self.assertEqual(test_calls, ["test"])
        self.assertEqual(another_calls, [42])
        
    def test_unsubscribe(self):
        """Test unsubscribe functionality."""
        calls = []
        
        def handler(event):
            calls.append(event.data)
            
        # Subscribe and verify it works
        self.event_bus.subscribe(TestEvent, handler)
        self.event_bus.publish(TestEvent("first"))
        self.assertEqual(calls, ["first"])
        
        # Unsubscribe
        result = self.event_bus.unsubscribe(TestEvent, handler)
        self.assertTrue(result)
        
        # Verify handler is no longer called
        self.event_bus.publish(TestEvent("second"))
        self.assertEqual(calls, ["first"])  # Should not have "second"
        
    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing a handler that wasn't subscribed."""
        def handler(event):
            pass
            
        result = self.event_bus.unsubscribe(TestEvent, handler)
        self.assertFalse(result)
        
    def test_inheritance_handling(self):
        """Test that handlers for parent event types receive derived events."""
        base_calls = []
        derived_calls = []
        
        def base_handler(event):
            base_calls.append(event.data)
            
        def derived_handler(event):
            derived_calls.append((event.data, event.extra))
            
        # Subscribe to both base and derived event types
        self.event_bus.subscribe(TestEvent, base_handler)
        self.event_bus.subscribe(DerivedTestEvent, derived_handler)
        
        # Publish derived event
        self.event_bus.publish(DerivedTestEvent("test", "extra"))
        
        # Both handlers should be called
        self.assertEqual(base_calls, ["test"])
        self.assertEqual(derived_calls, [("test", "extra")])
        
    def test_exception_in_handler(self):
        """Test that exceptions in handlers don't affect other handlers."""
        calls = []
        
        def failing_handler(event):
            raise RuntimeError("Handler error")
            
        def good_handler(event):
            calls.append(event.data)
            
        # Subscribe both handlers
        self.event_bus.subscribe(TestEvent, failing_handler)
        self.event_bus.subscribe(TestEvent, good_handler)
        
        # Publish event - should not raise
        self.event_bus.publish(TestEvent("test"))
        
        # Good handler should still be called
        self.assertEqual(calls, ["test"])
        
    def test_clear_subscriptions(self):
        """Test clearing all subscriptions."""
        calls = []
        
        def handler(event):
            calls.append(event.data)
            
        # Subscribe and verify
        self.event_bus.subscribe(TestEvent, handler)
        self.assertEqual(self.event_bus.get_subscription_count(), 1)
        
        # Clear and verify
        self.event_bus.clear_subscriptions()
        self.assertEqual(self.event_bus.get_subscription_count(), 0)
        
        # Handler should not be called
        self.event_bus.publish(TestEvent("test"))
        self.assertEqual(calls, [])
        
    def test_clear_subscriptions_for_type(self):
        """Test clearing subscriptions for specific event type."""
        test_calls = []
        another_calls = []
        
        def test_handler(event):
            test_calls.append(event.data)
            
        def another_handler(event):
            another_calls.append(event.value)
            
        # Subscribe to different event types
        self.event_bus.subscribe(TestEvent, test_handler)
        self.event_bus.subscribe(AnotherTestEvent, another_handler)
        
        # Clear only TestEvent subscriptions
        self.event_bus.clear_subscriptions_for_type(TestEvent)
        
        # Publish both events
        self.event_bus.publish(TestEvent("test"))
        self.event_bus.publish(AnotherTestEvent(42))
        
        # Only another_handler should be called
        self.assertEqual(test_calls, [])
        self.assertEqual(another_calls, [42])
        
    def test_get_subscription_count(self):
        """Test getting subscription count."""
        def handler1(event): pass
        def handler2(event): pass
        
        # Initial count
        self.assertEqual(self.event_bus.get_subscription_count(), 0)
        
        # Add subscriptions
        self.event_bus.subscribe(TestEvent, handler1)
        self.assertEqual(self.event_bus.get_subscription_count(), 1)
        self.assertEqual(self.event_bus.get_subscription_count(TestEvent), 1)
        
        self.event_bus.subscribe(TestEvent, handler2)
        self.assertEqual(self.event_bus.get_subscription_count(), 2)
        self.assertEqual(self.event_bus.get_subscription_count(TestEvent), 2)
        
        self.event_bus.subscribe(AnotherTestEvent, handler1)
        self.assertEqual(self.event_bus.get_subscription_count(), 3)
        self.assertEqual(self.event_bus.get_subscription_count(AnotherTestEvent), 1)
        
    def test_thread_safety(self):
        """Test thread safety of event bus operations."""
        results = []
        errors = []
        
        def handler(event):
            results.append(event.data)
            
        def subscribe_thread():
            try:
                for i in range(100):
                    self.event_bus.subscribe(TestEvent, handler)
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(('subscribe', e))
                
        def publish_thread():
            try:
                for i in range(100):
                    self.event_bus.publish(TestEvent(f"event_{i}"))
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(('publish', e))
                
        def unsubscribe_thread():
            try:
                for i in range(50):
                    time.sleep(0.0002)
                    self.event_bus.unsubscribe(TestEvent, handler)
            except Exception as e:
                errors.append(('unsubscribe', e))
                
        # Start threads
        threads = [
            threading.Thread(target=subscribe_thread),
            threading.Thread(target=publish_thread),
            threading.Thread(target=unsubscribe_thread)
        ]
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should have no errors
        self.assertEqual(errors, [])
        # Should have received some events
        self.assertGreater(len(results), 0)
        
    def test_type_validation(self):
        """Test type validation for publish and subscribe."""
        def handler(event): pass
        
        # Test publish with non-Event
        with self.assertRaises(TypeError):
            self.event_bus.publish("not an event")
            
        # Test subscribe with non-Event type
        with self.assertRaises(TypeError):
            self.event_bus.subscribe(str, handler)
            
        # Test unsubscribe with non-Event type
        with self.assertRaises(TypeError):
            self.event_bus.unsubscribe(str, handler)
            
        # Test clear_subscriptions_for_type with non-Event type
        with self.assertRaises(TypeError):
            self.event_bus.clear_subscriptions_for_type(str)
            
        # Test get_subscription_count with non-Event type
        with self.assertRaises(TypeError):
            self.event_bus.get_subscription_count(str)
            
    def test_same_handler_multiple_times(self):
        """Test subscribing same handler multiple times."""
        calls = []
        
        def handler(event):
            calls.append(event.data)
            
        # Subscribe same handler multiple times
        self.event_bus.subscribe(TestEvent, handler)
        self.event_bus.subscribe(TestEvent, handler)
        self.event_bus.subscribe(TestEvent, handler)
        
        # Should only be added once (set behavior)
        self.assertEqual(self.event_bus.get_subscription_count(TestEvent), 1)
        
        # Publish event
        self.event_bus.publish(TestEvent("test"))
        
        # Handler should only be called once
        self.assertEqual(calls, ["test"])


if __name__ == '__main__':
    unittest.main()