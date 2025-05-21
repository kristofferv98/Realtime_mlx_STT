"""
ProgressBar package for managing tqdm progress bar visibility.

This package automatically disables tqdm progress bars across the entire application
by setting environment variables before tqdm is imported anywhere.
"""
import os
import sys

# Set environment variables to disable progress bars
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Import ProgressBarManager - will be used when explicitly imported
from .ProgressBarManager import ProgressBarManager

# Try to use monkeypatching as a backup method
try:
    # Pre-emptively patch tqdm if it's already imported
    if 'tqdm' in sys.modules:
        import tqdm
        tqdm.tqdm.disable = True
        
        # Also try to patch tqdm.auto if it exists
        if hasattr(tqdm, 'auto'):
            tqdm.auto.tqdm.disable = True
except Exception:
    # If patching fails, that's okay - environment variables should still work
    pass

__all__ = ['ProgressBarManager']