"""
ProgressBarManager - Centralized management of tqdm progress bars across the application.

This module provides a centralized way to control the visibility of tqdm progress bars
throughout the application, including those from third-party libraries like Hugging Face Hub.
"""

import os
import sys
import tqdm
from src.Infrastructure.Logging import LoggingModule

# Set environment variables to disable progress bars
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Try to directly set tqdm.disable globally
tqdm.tqdm.disable = True

# Get a logger instance for this module
logger = LoggingModule.get_logger(__name__)

class ProgressBarManager:
    """
    Manager class for globally controlling tqdm progress bar visibility.
    
    This class provides static methods to enable/disable tqdm progress bars
    across the entire application, including those used by third-party libraries
    like Hugging Face Hub.
    """
    
    _initialized = False
    _disabled = False
    
    @classmethod
    def initialize(cls, disabled=None):
        """
        Initialize progress bar settings globally.
        
        Args:
            disabled (bool, optional): Explicitly set whether progress bars should be disabled.
                If None, will check environment variables.
                Defaults to None.
        
        Returns:
            bool: The current disabled state of progress bars after initialization.
        """
        # Check environment variables if parameter not provided
        env_disabled = (
            os.environ.get("DISABLE_PROGRESS_BARS", "").lower() in ("true", "1", "yes") or
            os.environ.get("TQDM_DISABLE", "").lower() in ("true", "1", "yes") or
            os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS", "").lower() in ("true", "1", "yes") or
            "TQDM_DISABLE" in os.environ or  # Check for presence regardless of value
            "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ  # Check for presence regardless of value
        )
        
        # Priority: explicit parameter > environment variables > default (enabled)
        cls._disabled = disabled if disabled is not None else env_disabled
        
        # Configure tqdm globally - use multiple approaches to ensure it works
        try:
            # Set the global tqdm disable flag
            tqdm.tqdm.disable = cls._disabled
            
            # Also try to access the class directly in case of different import patterns
            import tqdm as tqdm_module
            tqdm_module.tqdm.disable = cls._disabled
            
            # Try to access via auto module if it exists
            if hasattr(tqdm, 'auto'):
                tqdm.auto.tqdm.disable = cls._disabled
            
            # Ensure lock is initialized for thread safety
            try:
                tqdm.tqdm.set_lock(tqdm.tqdm.get_lock())
            except:
                # If lock initialization fails, it's not critical
                pass
                
        except Exception as e:
            logger.warning(f"Error configuring tqdm: {e}")
            # Continue anyway, as we'll still use environment variables as fallback
        
        # Disable HuggingFace progress bars if needed
        if cls._disabled:
            # Set environment variables for both tqdm and huggingface-hub
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TQDM_DISABLE"] = "1"  # Alternative way some libraries check
            logger.debug("Progress bars globally disabled")
        else:
            # Remove the environment variables if they exist
            if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS")
            if "TQDM_DISABLE" in os.environ:
                os.environ.pop("TQDM_DISABLE")
            logger.debug("Progress bars globally enabled")
        
        cls._initialized = True
        return cls._disabled
    
    @classmethod
    def disable(cls):
        """
        Disable progress bars globally.
        
        Returns:
            bool: Always returns True to indicate progress bars are disabled.
        """
        if not cls._initialized:
            cls.initialize()
        
        cls._disabled = True
        tqdm.tqdm.disable = True
        # Set environment variables for both tqdm and huggingface-hub
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TQDM_DISABLE"] = "1"  # Alternative way some libraries check
        logger.debug("Progress bars explicitly disabled")
        return True
    
    @classmethod
    def enable(cls):
        """
        Enable progress bars globally.
        
        Returns:
            bool: Always returns False to indicate progress bars are enabled.
        """
        if not cls._initialized:
            cls.initialize()
        
        cls._disabled = False
        tqdm.tqdm.disable = False
        
        # Remove the environment variables if they exist
        if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS")
        if "TQDM_DISABLE" in os.environ:
            os.environ.pop("TQDM_DISABLE")
        
        logger.debug("Progress bars explicitly enabled")
        return False
    
    @classmethod
    def is_disabled(cls):
        """
        Check if progress bars are disabled.
        
        Returns:
            bool: True if progress bars are disabled, False otherwise.
        """
        if not cls._initialized:
            cls.initialize()
        
        return cls._disabled