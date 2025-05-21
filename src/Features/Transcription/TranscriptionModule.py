"""
TranscriptionModule for registering and providing access to transcription functionality.

This module serves as the main entry point for the Transcription feature,
handling registration of commands, handlers, and providing a public API.
"""

import uuid
from typing import Dict, List, Any, Optional, Callable, Union
import numpy as np

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import IEventBus

# Feature-specific imports
from src.Features.Transcription.Commands.TranscribeAudioCommand import TranscribeAudioCommand
from src.Features.Transcription.Commands.ConfigureTranscriptionCommand import ConfigureTranscriptionCommand
from src.Features.Transcription.Commands.StartTranscriptionSessionCommand import StartTranscriptionSessionCommand
from src.Features.Transcription.Commands.StopTranscriptionSessionCommand import StopTranscriptionSessionCommand

from src.Features.Transcription.Events.TranscriptionStartedEvent import TranscriptionStartedEvent
from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
from src.Features.Transcription.Events.TranscriptionErrorEvent import TranscriptionErrorEvent

from src.Features.Transcription.Handlers.TranscriptionCommandHandler import TranscriptionCommandHandler

from src.Features.Transcription.Models.TranscriptionResult import TranscriptionResult


class TranscriptionModule:
    """
    Module for speech-to-text transcription functionality.
    
    This class provides registration and convenience methods for the Transcription feature.
    It serves as a facade for the underlying components (engines, handlers, etc.).
    """
    
    @staticmethod
    def register(command_dispatcher: CommandDispatcher,
                event_bus: IEventBus,
                default_engine: str = "mlx_whisper",
                default_model: str = "whisper-large-v3-turbo",
                default_language: Optional[str] = None,
                openai_api_key: Optional[str] = None) -> TranscriptionCommandHandler:
        """
        Register the Transcription feature with the system.
        
        This method:
        1. Creates the transcription handler
        2. Registers it with the command dispatcher
        3. Configures the default engine
        
        Args:
            command_dispatcher: The command dispatcher to register handlers with
            event_bus: The event bus for publishing/subscribing to events
            default_engine: The default engine to use ('mlx_whisper' or 'openai')
            default_model: The default model to use ('whisper-large-v3-turbo', 'gpt-4o-transcribe', etc.)
            default_language: Default language code or None for auto-detection
            openai_api_key: API key for OpenAI services (required if using 'openai' engine)
            
        Returns:
            TranscriptionCommandHandler: The registered command handler
        """
        logger = LoggingModule.get_logger(__name__)
        logger.info("Registering Transcription feature")
        
        # Create command handler
        handler = TranscriptionCommandHandler(event_bus=event_bus)
        
        # Register with command dispatcher
        command_dispatcher.register_handler(TranscribeAudioCommand, handler)
        command_dispatcher.register_handler(ConfigureTranscriptionCommand, handler)
        command_dispatcher.register_handler(StartTranscriptionSessionCommand, handler)
        command_dispatcher.register_handler(StopTranscriptionSessionCommand, handler)
        
        # Configure default engine
        try:
            # Set up configuration
            config_dict = {
                "engine_type": default_engine,
                "model_name": default_model,
                "language": default_language
            }
            
            # Add OpenAI API key if using the OpenAI engine
            if default_engine == "openai" and openai_api_key:
                config_dict["openai_api_key"] = openai_api_key
            
            config_command = ConfigureTranscriptionCommand(**config_dict)
            result = command_dispatcher.dispatch(config_command)
            
            if result:
                logger.info(f"Successfully configured {default_engine} engine with model {default_model}")
            else:
                logger.warning(f"Failed to configure {default_engine} engine")
                
                # Check for common configuration issues
                if default_engine == "openai" and not openai_api_key:
                    logger.error("OpenAI engine requires an API key. Provide one via openai_api_key parameter or OPENAI_API_KEY environment variable.")
            
        except Exception as e:
            logger.error(f"Error configuring default engine: {e}", exc_info=True)
        
        return handler
    
    @staticmethod
    def configure(command_dispatcher: CommandDispatcher,
                 engine_type: str = "mlx_whisper",
                 model_name: str = "whisper-large-v3-turbo",
                 language: Optional[str] = None,
                 streaming: bool = True,
                 openai_api_key: Optional[str] = None,
                 **kwargs) -> bool:
        """
        Configure the transcription system.
        
        Args:
            command_dispatcher: The command dispatcher to use
            engine_type: Type of transcription engine to use ('mlx_whisper' or 'openai')
            model_name: Name of the model to use ('whisper-large-v3-turbo', 'gpt-4o-transcribe', etc.)
            language: Optional language code or None for auto-detection
            streaming: Whether to enable streaming mode
            openai_api_key: Optional API key for OpenAI services (required if using 'openai' engine)
            **kwargs: Additional engine-specific parameters
            
        Returns:
            bool: True if configuration was successful
        """
        # Create command parameters
        command_params = {
            "engine_type": engine_type,
            "model_name": model_name,
            "language": language,
            "streaming": streaming
        }
        
        # Add OpenAI API key if provided
        if engine_type == "openai" and openai_api_key:
            command_params["openai_api_key"] = openai_api_key
            
        # Add any additional options
        command_params["options"] = kwargs
        
        command = ConfigureTranscriptionCommand(**command_params)
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def start_session(command_dispatcher: CommandDispatcher,
                     session_id: Optional[str] = None,
                     language: Optional[str] = None,
                     streaming: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Start a new transcription session.
        
        Args:
            command_dispatcher: The command dispatcher to use
            session_id: Optional session ID (auto-generated if None)
            language: Optional language code or None for auto-detection
            streaming: Whether to use streaming mode
            **kwargs: Additional session configuration parameters
            
        Returns:
            Dict[str, Any]: Session information with session_id
        """
        command = StartTranscriptionSessionCommand(
            session_id=session_id or str(uuid.uuid4()),
            language=language,
            streaming=streaming,
            config=kwargs
        )
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def stop_session(command_dispatcher: CommandDispatcher,
                    session_id: str,
                    flush_remaining_audio: bool = True,
                    save_results: bool = False,
                    output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop a transcription session.
        
        Args:
            command_dispatcher: The command dispatcher to use
            session_id: ID of the session to stop
            flush_remaining_audio: Whether to process any remaining audio
            save_results: Whether to save the results
            output_path: Path to save the results to
            
        Returns:
            Dict[str, Any]: Operation result
        """
        command = StopTranscriptionSessionCommand(
            session_id=session_id,
            flush_remaining_audio=flush_remaining_audio,
            save_results=save_results,
            output_path=output_path
        )
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def transcribe_audio(command_dispatcher: CommandDispatcher,
                        audio_data: np.ndarray,
                        session_id: Optional[str] = None,
                        is_first_chunk: bool = False,
                        is_last_chunk: bool = False,
                        language: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio data.
        
        Args:
            command_dispatcher: The command dispatcher to use
            audio_data: Audio data as numpy array
            session_id: Optional session ID (auto-generated if None)
            is_first_chunk: Whether this is the first chunk in a session
            is_last_chunk: Whether this is the final chunk in a session
            language: Optional language code or None for auto-detection
            **kwargs: Additional transcription options
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        # Create a new session if needed
        if session_id is None:
            session_result = TranscriptionModule.start_session(
                command_dispatcher,
                language=language,
                **kwargs
            )
            session_id = session_result.get("session_id")
            is_first_chunk = True
        
        # Create command
        command = TranscribeAudioCommand(
            audio_chunk=audio_data,
            session_id=session_id,
            is_first_chunk=is_first_chunk,
            is_last_chunk=is_last_chunk,
            language=language,
            options=kwargs
        )
        
        # Dispatch command
        result = command_dispatcher.dispatch(command)
        
        # Close session if this was the last chunk
        if is_last_chunk:
            TranscriptionModule.stop_session(
                command_dispatcher,
                session_id=session_id,
                flush_remaining_audio=False  # We've already processed this as the last chunk
            )
        
        return result
    
    @staticmethod
    def transcribe_file(command_dispatcher: CommandDispatcher,
                       file_path: str,
                       language: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            command_dispatcher: The command dispatcher to use
            file_path: Path to the audio file to transcribe
            language: Optional language code or None for auto-detection
            **kwargs: Additional transcription options
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        logger = LoggingModule.get_logger(__name__)
        logger.info(f"Transcribing file: {file_path}")
        
        try:
            # Create a new session
            # Remove streaming from kwargs if it exists to avoid duplicates
            session_kwargs = kwargs.copy()
            if 'streaming' in session_kwargs:
                del session_kwargs['streaming']
                
            # Store timeout separately - don't pass it to session config
            transcription_timeout = kwargs.get('timeout', 60.0)  # 60 seconds timeout for files
                
            session_result = TranscriptionModule.start_session(
                command_dispatcher,
                language=language,
                streaming=False,  # Use batch mode for files
                **session_kwargs
            )
            
            # Check if we got a valid response with a session_id
            session_id = None
            if isinstance(session_result, dict) and "session_id" in session_result:
                session_id = session_result.get("session_id")
            elif isinstance(session_result, (list, tuple)) and len(session_result) > 0:
                # Try to get first item if it's a list
                if isinstance(session_result[0], dict) and "session_id" in session_result[0]:
                    session_id = session_result[0].get("session_id")
            
            # Generate a fallback session ID if needed
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.warning(f"Could not get session_id from result, using generated ID: {session_id}")
            
            logger.info(f"Created transcription session: {session_id}")
            
            # First, try direct file path transcription
            # This path lets the lower-level engine handle file loading which can be more optimized
            try:
                logger.info(f"Attempting direct file transcription of {file_path}")
                
                # Transcribe using the file path directly
                result = TranscriptionModule.transcribe_audio(
                    command_dispatcher,
                    audio_data=file_path,  # Pass the file path directly
                    session_id=session_id,
                    is_first_chunk=True,
                    is_last_chunk=True,
                    language=language,
                    file_mode=True,  # Flag for file transcription
                    **kwargs
                )
                
                if not isinstance(result, dict) or result.get("error"):
                    logger.warning(f"Direct file transcription failed, falling back to audio loading: {result}")
                    raise ValueError("Direct file transcription failed")
                    
                return result
                
            except Exception as direct_error:
                # Log the error but don't fail yet, try the fallback approach
                logger.warning(f"Direct file transcription failed: {direct_error}")
                logger.info("Falling back to loading audio file ourselves...")
                
                # Load audio file
                try:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(file_path, dtype='float32')
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        import librosa
                        audio_data = librosa.resample(
                            audio_data,
                            orig_sr=sample_rate,
                            target_sr=16000
                        )
                    
                    # Ensure mono
                    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Normalize
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val
                        
                    logger.info(f"Successfully loaded audio: {len(audio_data)/16000:.2f}s at {sample_rate}Hz")
                
                except Exception as load_error:
                    logger.error(f"Could not load audio file: {load_error}")
                    return {"error": f"Audio loading failed: {str(load_error)}"}
            
                # Transcribe as a single chunk
                result = TranscriptionModule.transcribe_audio(
                    command_dispatcher,
                    audio_data=audio_data,
                    session_id=session_id,
                    is_first_chunk=True,
                    is_last_chunk=True,
                    language=language,
                    **kwargs
                )
                
                return result
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}", exc_info=True)
            return {"error": str(e)}
    
    @staticmethod
    def on_transcription_started(event_bus: IEventBus,
                               handler: Callable[[str, Optional[str], float], None]) -> None:
        """
        Subscribe to transcription started events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when transcription starts
                    Function receives (session_id, language, timestamp)
        """
        event_bus.subscribe(TranscriptionStartedEvent, 
                           lambda event: handler(
                               event.session_id,
                               event.language,
                               event.audio_timestamp
                           ))
    
    @staticmethod
    def on_transcription_updated(event_bus: IEventBus,
                               handler: Callable[[str, str, bool, float], None]) -> None:
        """
        Subscribe to transcription updated events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when transcription text is updated
                    Function receives (session_id, text, is_final, confidence)
        """
        event_bus.subscribe(TranscriptionUpdatedEvent,
                           lambda event: handler(
                               event.session_id,
                               event.text,
                               event.is_final,
                               event.confidence
                           ))
    
    @staticmethod
    def on_transcription_error(event_bus: IEventBus,
                              handler: Callable[[str, str, str], None]) -> None:
        """
        Subscribe to transcription error events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when transcription encounters an error
                    Function receives (session_id, error_message, error_type)
        """
        event_bus.subscribe(TranscriptionErrorEvent,
                           lambda event: handler(
                               event.session_id,
                               event.error_message,
                               event.error_type
                           ))
    
    @staticmethod
    def register_vad_integration(event_bus: IEventBus, 
                                transcription_handler: TranscriptionCommandHandler,
                                session_id: Optional[str] = None,
                                auto_start_on_speech: bool = True) -> None:
        """
        Register integration with Voice Activity Detection.
        
        This method subscribes to VAD events to automatically transcribe complete
        speech segments when silence is detected following speech.
        
        Args:
            event_bus: The event bus to subscribe to
            transcription_handler: The transcription command handler to use
            session_id: Optional session ID to use for all VAD-triggered transcriptions 
                      (if None, each speech segment gets its own session)
            auto_start_on_speech: Whether to automatically start a new transcription 
                                 session when speech is detected
        """
        logger = LoggingModule.get_logger(__name__)
        logger.info("Registering VAD integration for transcription")
        
        # Import VAD events
        from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
        from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
        
        # Create a session map to track active VAD-triggered sessions
        speech_sessions = {}
        
        # Handle silence detected events
        def on_silence_detected(event):
            # Check if audio_reference exists and has content
            if event.audio_reference is None:
                logger.warning("Received SilenceDetectedEvent with no audio_reference")
                return
                
            # Check if it's a numpy array with content
            if isinstance(event.audio_reference, np.ndarray) and event.audio_reference.size == 0:
                logger.warning("Received SilenceDetectedEvent with empty numpy array")
                return
                
            speech_id = event.speech_id
            speech_session_id = speech_sessions.get(speech_id, session_id or speech_id)
            
            logger.info(f"Processing complete speech segment (id: {speech_id}, "
                       f"duration: {event.speech_duration:.2f}s)")
            
            # Process the complete speech segment
            try:
                transcription_handler.on_silence_detected(
                    session_id=speech_session_id,
                    audio_reference=event.audio_reference,
                    duration=event.speech_duration
                )
                
                # Clean up session tracking if using individual sessions
                if session_id is None and speech_id in speech_sessions:
                    del speech_sessions[speech_id]
                    
            except Exception as e:
                logger.error(f"Error processing VAD-triggered transcription: {e}", exc_info=True)
        
        # Handle speech detected events if auto-start is enabled
        def on_speech_detected(event):
            if not auto_start_on_speech:
                return
                
            speech_id = event.speech_id
            
            # If using a global session, don't need to do anything special
            if session_id is not None:
                return
                
            # Create a new session for this speech segment
            try:
                # Generate a unique session ID for this speech segment
                speech_session_id = f"vad-{speech_id}"
                speech_sessions[speech_id] = speech_session_id
                
                logger.info(f"VAD detected new speech (id: {speech_id}), "
                           f"preparing transcription session {speech_session_id}")
                
            except Exception as e:
                logger.error(f"Error setting up VAD-triggered transcription session: {e}", 
                            exc_info=True)
        
        # Subscribe to the events
        event_bus.subscribe(SilenceDetectedEvent, on_silence_detected)
        
        if auto_start_on_speech:
            event_bus.subscribe(SpeechDetectedEvent, on_speech_detected)