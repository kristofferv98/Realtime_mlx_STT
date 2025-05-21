#!/bin/bash
# Script to set logging environment variables

# Check if we're setting development or production environment
env_type=${1:-dev}

if [ "$env_type" = "dev" ] || [ "$env_type" = "development" ]; then
    echo "Setting logging environment variables for development..."
    
    # Development settings - verbose logging
    export REALTIMESTT_LOG_LEVEL=DEBUG
    export REALTIMESTT_LOG_CONSOLE=true
    export REALTIMESTT_LOG_FILE=true
    export REALTIMESTT_LOG_PATH=logs/realtimestt_dev.log
    export REALTIMESTT_LOG_ROTATION=true
    
    # Runtime control server settings
    export REALTIMESTT_LOG_CONTROL_SERVER=true
    export REALTIMESTT_LOG_CONTROL_PORT=50101
    
    # Progress bar settings - show progress bars in development for debugging
    # unset DISABLE_PROGRESS_BARS
    unset TQDM_DISABLE
    unset HF_HUB_DISABLE_PROGRESS_BARS
    
    # Feature-specific settings
    export REALTIMESTT_LOG_FEATURE_AUDIOCAPTURE=DEBUG
    export REALTIMESTT_LOG_FEATURE_TRANSCRIPTION=DEBUG
    export REALTIMESTT_LOG_FEATURE_VOICEACTIVITYDETECTION=DEBUG
    export REALTIMESTT_LOG_FEATURE_WAKEWORDDETECTION=DEBUG
    
elif [ "$env_type" = "prod" ] || [ "$env_type" = "production" ]; then
    echo "Setting logging environment variables for production..."
    
    # Production settings - focused on important info
    export REALTIMESTT_LOG_LEVEL=INFO
    export REALTIMESTT_LOG_CONSOLE=true
    export REALTIMESTT_LOG_FILE=true
    export REALTIMESTT_LOG_PATH=logs/realtimestt.log
    export REALTIMESTT_LOG_ROTATION=true
    
    # Runtime control server settings - disabled by default in production
    export REALTIMESTT_LOG_CONTROL_SERVER=false
    export REALTIMESTT_LOG_CONTROL_PORT=50101
    
    # Progress bar settings - hide progress bars in production for cleaner output
    export DISABLE_PROGRESS_BARS=true
    export TQDM_DISABLE=1 
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    
    # Feature-specific settings
    export REALTIMESTT_LOG_FEATURE_AUDIOCAPTURE=INFO
    export REALTIMESTT_LOG_FEATURE_TRANSCRIPTION=INFO
    export REALTIMESTT_LOG_FEATURE_VOICEACTIVITYDETECTION=INFO
    export REALTIMESTT_LOG_FEATURE_WAKEWORDDETECTION=INFO
    
else
    echo "Setting custom logging environment variables..."
    
    # Custom settings based on argument
    export REALTIMESTT_LOG_LEVEL=INFO
    export REALTIMESTT_LOG_CONSOLE=true
    export REALTIMESTT_LOG_FILE=true
    export REALTIMESTT_LOG_PATH=logs/realtimestt_custom.log
    export REALTIMESTT_LOG_ROTATION=true
    
    # Runtime control server settings
    export REALTIMESTT_LOG_CONTROL_SERVER=true
    export REALTIMESTT_LOG_CONTROL_PORT=50101
    
    # Progress bar settings - hide progress bars by default in custom environments
    # Can be overridden by setting variables directly in the environment
    export DISABLE_PROGRESS_BARS=true
    export TQDM_DISABLE=1
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    
    # Set a specific feature to debug level if specified
    if [ ! -z "$1" ]; then
        feature=$(echo "$1" | tr '[:lower:]' '[:upper:]')
        echo "Setting debug level for feature: $1"
        export "REALTIMESTT_LOG_FEATURE_$feature=DEBUG"
    fi
fi

echo "Logging environment variables set. Run your script in this shell session to use these settings."
echo "For example: python examples/wake_word_detection.py"