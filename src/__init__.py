"""
Realtime_mlx_STT package.

This is the main package for the Realtime_mlx_STT library.
"""
import os

# Set environment variables to disable progress bars globally
# This ensures they're set as early as possible in the import chain
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Let's avoid potential circular imports
# We'll just set the environment variables here and let the ProgressBar module
# handle the patching when it's imported

# Make key modules available at the package level
from .Core import EventBus, CommandDispatcher
from .Infrastructure import LoggingModule

__all__ = [
    'EventBus',
    'CommandDispatcher',
    'LoggingModule'
]