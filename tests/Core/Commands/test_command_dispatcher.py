#!/usr/bin/env python3
"""
Unit tests for CommandDispatcher.

This module tests the command dispatcher implementation,
including command routing, handler registration, and error handling.
"""

import unittest
from unittest.mock import Mock, MagicMock
import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Commands.command import Command
from src.Core.Common.Interfaces.command_handler import ICommandHandler


# Test command classes
class TestCommand(Command):
    """Test command for unit tests."""
    def __init__(self, data: str):
        super().__init__(name="TestCommand")
        self.data = data


class AnotherTestCommand(Command):
    """Another test command for unit tests."""
    def __init__(self, value: int):
        super().__init__(name="AnotherTestCommand")
        self.value = value


class DerivedTestCommand(TestCommand):
    """Derived test command to test inheritance."""
    def __init__(self, data: str, extra: str):
        super().__init__(data)
        self.extra = extra


# Test handler classes
class TestHandler(ICommandHandler[str]):
    """Test handler that returns a string."""
    
    def __init__(self):
        self.handled_commands = []
        
    def handle(self, command: Command) -> str:
        self.handled_commands.append(command)
        if isinstance(command, TestCommand):
            return f"Handled: {command.data}"
        return "Handled"
        
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, TestCommand)


class AnotherTestHandler(ICommandHandler[int]):
    """Test handler that returns an int."""
    
    def __init__(self):
        self.handled_commands = []
        
    def handle(self, command: Command) -> int:
        self.handled_commands.append(command)
        if isinstance(command, AnotherTestCommand):
            return command.value * 2
        return 0
        
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, AnotherTestCommand)


class UniversalTestHandler(ICommandHandler[Any]):
    """Test handler that can handle any command."""
    
    def __init__(self):
        self.handled_commands = []
        
    def handle(self, command: Command) -> Any:
        self.handled_commands.append(command)
        return command.name
        
    def can_handle(self, command: Command) -> bool:
        return True


class SelectiveTestHandler(ICommandHandler[str]):
    """Test handler that only handles commands with specific data."""
    
    def __init__(self, accept_data: str):
        self.accept_data = accept_data
        self.handled_commands = []
        
    def handle(self, command: Command) -> str:
        self.handled_commands.append(command)
        return f"Selective: {command.data}"
        
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, TestCommand) and command.data == self.accept_data


class TestCommandDispatcher(unittest.TestCase):
    """Test cases for CommandDispatcher."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = CommandDispatcher()
        
    def tearDown(self):
        """Clean up after tests."""
        self.dispatcher.clear_handlers()
        
    def test_initialization(self):
        """Test command dispatcher initialization."""
        dispatcher = CommandDispatcher()
        self.assertIsNotNone(dispatcher)
        self.assertEqual(dispatcher.get_handler_count(), 0)
        
    def test_register_handler(self):
        """Test handler registration."""
        handler = TestHandler()
        
        # Register handler
        self.dispatcher.register_handler(TestCommand, handler)
        
        # Verify registration
        self.assertEqual(self.dispatcher.get_handler_count(), 1)
        self.assertEqual(self.dispatcher.get_handler_count(TestCommand), 1)
        
    def test_dispatch_single_handler(self):
        """Test dispatching to a single handler."""
        handler = TestHandler()
        self.dispatcher.register_handler(TestCommand, handler)
        
        # Dispatch command
        command = TestCommand("test data")
        results = self.dispatcher.dispatch(command)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Handled: test data")
        self.assertEqual(len(handler.handled_commands), 1)
        self.assertEqual(handler.handled_commands[0], command)
        
    def test_dispatch_multiple_handlers(self):
        """Test dispatching to multiple handlers."""
        handler1 = TestHandler()
        handler2 = UniversalTestHandler()
        
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(TestCommand, handler2)
        
        # Dispatch command
        command = TestCommand("test")
        results = self.dispatcher.dispatch(command)
        
        # Verify both handlers were called
        self.assertEqual(len(results), 2)
        self.assertIn("Handled: test", results)
        self.assertIn("TestCommand", results)
        
    def test_dispatch_to_first_handler(self):
        """Test dispatch_to_first_handler method."""
        handler1 = TestHandler()
        handler2 = UniversalTestHandler()
        
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(TestCommand, handler2)
        
        # Dispatch to first handler
        command = TestCommand("test")
        result = self.dispatcher.dispatch_to_first_handler(command)
        
        # Should only get result from first handler
        self.assertEqual(result, "Handled: test")
        self.assertEqual(len(handler1.handled_commands), 1)
        self.assertEqual(len(handler2.handled_commands), 0)  # Second handler not called
        
    def test_selective_handling(self):
        """Test handlers that selectively handle commands."""
        handler1 = SelectiveTestHandler("accept")
        handler2 = SelectiveTestHandler("reject")
        
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(TestCommand, handler2)
        
        # Dispatch command that only handler1 accepts
        command = TestCommand("accept")
        results = self.dispatcher.dispatch(command)
        
        # Only handler1 should handle it
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Selective: accept")
        self.assertEqual(len(handler1.handled_commands), 1)
        self.assertEqual(len(handler2.handled_commands), 0)
        
    def test_different_command_types(self):
        """Test different command types are routed correctly."""
        test_handler = TestHandler()
        another_handler = AnotherTestHandler()
        
        self.dispatcher.register_handler(TestCommand, test_handler)
        self.dispatcher.register_handler(AnotherTestCommand, another_handler)
        
        # Dispatch different command types
        test_cmd = TestCommand("test")
        another_cmd = AnotherTestCommand(21)
        
        results1 = self.dispatcher.dispatch(test_cmd)
        results2 = self.dispatcher.dispatch(another_cmd)
        
        # Verify correct routing
        self.assertEqual(results1, ["Handled: test"])
        self.assertEqual(results2, [42])  # 21 * 2
        
    def test_inheritance_handling(self):
        """Test that handlers for parent command types receive derived commands."""
        handler = TestHandler()
        self.dispatcher.register_handler(TestCommand, handler)
        
        # Dispatch derived command
        command = DerivedTestCommand("test", "extra")
        results = self.dispatcher.dispatch(command)
        
        # Parent handler should handle it
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Handled: test")
        
    def test_no_handlers_registered(self):
        """Test dispatching when no handlers are registered."""
        command = TestCommand("test")
        
        with self.assertRaises(ValueError) as context:
            self.dispatcher.dispatch(command)
            
        self.assertIn("No handlers registered", str(context.exception))
        
    def test_no_matching_handlers(self):
        """Test when no handlers can handle the command."""
        handler = SelectiveTestHandler("specific")
        self.dispatcher.register_handler(TestCommand, handler)
        
        # Command that handler won't accept
        command = TestCommand("other")
        
        # dispatch returns empty list when no handlers match
        results = self.dispatcher.dispatch(command)
        self.assertEqual(results, [])
        
        # dispatch_to_first_handler raises when no handlers match
        with self.assertRaises(ValueError) as context:
            self.dispatcher.dispatch_to_first_handler(command)
            
        self.assertIn("No handler can handle command", str(context.exception))
        
    def test_handler_exception(self):
        """Test that exceptions in handlers are propagated."""
        # Create mock handler that raises exception
        handler = Mock(spec=ICommandHandler)
        handler.can_handle.return_value = True
        handler.handle.side_effect = RuntimeError("Handler error")
        
        self.dispatcher.register_handler(TestCommand, handler)
        
        # Should raise the handler's exception
        with self.assertRaises(RuntimeError) as context:
            self.dispatcher.dispatch(TestCommand("test"))
            
        self.assertEqual(str(context.exception), "Handler error")
        
    def test_unregister_handler(self):
        """Test unregistering handlers."""
        handler = TestHandler()
        
        # Register and verify
        self.dispatcher.register_handler(TestCommand, handler)
        self.assertEqual(self.dispatcher.get_handler_count(), 1)
        
        # Unregister and verify
        result = self.dispatcher.unregister_handler(TestCommand, handler)
        self.assertTrue(result)
        self.assertEqual(self.dispatcher.get_handler_count(), 0)
        
        # Try to unregister again
        result = self.dispatcher.unregister_handler(TestCommand, handler)
        self.assertFalse(result)
        
    def test_clear_handlers(self):
        """Test clearing all handlers."""
        handler1 = TestHandler()
        handler2 = AnotherTestHandler()
        
        # Register multiple handlers
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(AnotherTestCommand, handler2)
        self.assertEqual(self.dispatcher.get_handler_count(), 2)
        
        # Clear all handlers
        self.dispatcher.clear_handlers()
        self.assertEqual(self.dispatcher.get_handler_count(), 0)
        
    def test_clear_handlers_for_type(self):
        """Test clearing handlers for specific command type."""
        handler1 = TestHandler()
        handler2 = AnotherTestHandler()
        
        # Register handlers for different command types
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(AnotherTestCommand, handler2)
        
        # Clear only TestCommand handlers
        self.dispatcher.clear_handlers_for_type(TestCommand)
        
        # Verify selective clearing
        self.assertEqual(self.dispatcher.get_handler_count(TestCommand), 0)
        self.assertEqual(self.dispatcher.get_handler_count(AnotherTestCommand), 1)
        self.assertEqual(self.dispatcher.get_handler_count(), 1)
        
    def test_type_validation(self):
        """Test type validation for commands and handlers."""
        handler = TestHandler()
        
        # Test dispatch with non-Command
        with self.assertRaises(TypeError):
            self.dispatcher.dispatch("not a command")
            
        # Test dispatch_to_first_handler with non-Command
        with self.assertRaises(TypeError):
            self.dispatcher.dispatch_to_first_handler("not a command")
            
        # Test register with non-Command type
        with self.assertRaises(TypeError):
            self.dispatcher.register_handler(str, handler)
            
        # Test register with non-ICommandHandler
        with self.assertRaises(TypeError):
            self.dispatcher.register_handler(TestCommand, "not a handler")
            
    def test_multiple_handlers_same_type(self):
        """Test registering multiple handlers for same command type."""
        handler1 = TestHandler()
        handler2 = UniversalTestHandler()
        handler3 = SelectiveTestHandler("test")
        
        # Register multiple handlers for same command type
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(TestCommand, handler2)
        self.dispatcher.register_handler(TestCommand, handler3)
        
        # Verify all are registered
        self.assertEqual(self.dispatcher.get_handler_count(TestCommand), 3)
        
        # Dispatch command
        command = TestCommand("test")
        results = self.dispatcher.dispatch(command)
        
        # All handlers that can handle should be called
        self.assertEqual(len(results), 3)
        
    def test_handler_order(self):
        """Test that handlers are called in registration order."""
        results = []
        
        # Create handlers that append to results
        handler1 = Mock(spec=ICommandHandler)
        handler1.can_handle.return_value = True
        handler1.handle.side_effect = lambda cmd: results.append(1) or "result1"
        
        handler2 = Mock(spec=ICommandHandler)
        handler2.can_handle.return_value = True
        handler2.handle.side_effect = lambda cmd: results.append(2) or "result2"
        
        handler3 = Mock(spec=ICommandHandler)
        handler3.can_handle.return_value = True
        handler3.handle.side_effect = lambda cmd: results.append(3) or "result3"
        
        # Register in specific order
        self.dispatcher.register_handler(TestCommand, handler1)
        self.dispatcher.register_handler(TestCommand, handler2)
        self.dispatcher.register_handler(TestCommand, handler3)
        
        # Dispatch command
        self.dispatcher.dispatch(TestCommand("test"))
        
        # Verify order
        self.assertEqual(results, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()