"""
TranscriptionCommandHandler for processing transcription commands.

This handler manages transcription sessions, processes audio data,
and publishes transcription events.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union, cast

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

# Core imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.Core.Common.Interfaces.command_handler import ICommandHandler
    from src.Core.Commands.command import Command
    from src.Core.Events.event_bus import IEventBus
    from src.Core.Common.Interfaces.transcription_engine import ITranscriptionEngine
else:
    from src.Core.Common.Interfaces.command_handler import ICommandHandler
    from src.Core.Commands.command import Command
    from src.Core.Events.event_bus import IEventBus
    from src.Core.Common.Interfaces.transcription_engine import ITranscriptionEngine

# Feature-specific imports
from src.Features.Transcription.Commands.TranscribeAudioCommand import TranscribeAudioCommand
from src.Features.Transcription.Commands.ConfigureTranscriptionCommand import ConfigureTranscriptionCommand
from src.Features.Transcription.Commands.StartTranscriptionSessionCommand import StartTranscriptionSessionCommand
from src.Features.Transcription.Commands.StopTranscriptionSessionCommand import StopTranscriptionSessionCommand

from src.Features.Transcription.Events.TranscriptionStartedEvent import TranscriptionStartedEvent
from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
from src.Features.Transcription.Events.TranscriptionErrorEvent import TranscriptionErrorEvent

from src.Features.Transcription.Models.TranscriptionSession import TranscriptionSession
from src.Features.Transcription.Models.TranscriptionConfig import TranscriptionConfig
from src.Features.Transcription.Models.TranscriptionResult import TranscriptionResult

# Import for Transcription Manager and engines
from src.Features.Transcription.Engines.DirectTranscriptionManager import DirectTranscriptionManager
from src.Features.Transcription.Engines.OpenAITranscriptionEngine import OpenAITranscriptionEngine


class TranscriptionCommandHandler(ICommandHandler[Any]):
    """
    Handler for transcription commands.
    
    This handler processes commands related to transcription, manages
    transcription sessions, and coordinates with the transcription engine.
    """
    
    def __init__(self, event_bus: IEventBus):
        """
        Initialize the transcription command handler.
        
        Args:
            event_bus: Event bus for publishing events
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.event_bus = event_bus
        
        # Active sessions with tracking for automatic cleanup
        self.sessions: Dict[str, TranscriptionSession] = {}
        self.session_last_activity: Dict[str, float] = {}
        
        # Session expiry configuration
        self.session_expiry_seconds = 600  # Default: expire sessions after 10 minutes of inactivity
        self.max_sessions = 100  # Maximum number of simultaneous sessions to prevent memory issues
        
        # Transcription manager - using Direct implementation
        self.transcription_manager = DirectTranscriptionManager()
        
        # Default configuration
        self.default_config = TranscriptionConfig()
        
        # Session cleanup counter
        self._cleanup_counter = 0
        self._cleanup_interval = 20  # Check for expired sessions every 20 operations
        
        self.logger.info("Initialized TranscriptionCommandHandler")
    
    def handle(self, command: Command) -> Any:
        """
        Handle a transcription command and produce a result.
        
        Args:
            command: The command to handle
            
        Returns:
            The result of the command execution (type depends on command)
            
        Raises:
            TypeError: If the command is not of the expected type
            ValueError: If the command contains invalid data
            Exception: If an error occurs during command execution
        """
        if isinstance(command, TranscribeAudioCommand):
            return self._handle_transcribe_audio(command)
        elif isinstance(command, ConfigureTranscriptionCommand):
            return self._handle_configure_transcription(command)
        elif isinstance(command, StartTranscriptionSessionCommand):
            return self._handle_start_session(command)
        elif isinstance(command, StopTranscriptionSessionCommand):
            return self._handle_stop_session(command)
        else:
            raise TypeError(f"Unsupported command type: {type(command).__name__}")
    
    def can_handle(self, command: Command) -> bool:
        """
        Check if this handler can handle the given command.
        
        Args:
            command: The command to check
            
        Returns:
            bool: True if this handler can handle the command, False otherwise
        """
        return isinstance(command, (
            TranscribeAudioCommand,
            ConfigureTranscriptionCommand,
            StartTranscriptionSessionCommand,
            StopTranscriptionSessionCommand
        ))
    
    def _handle_transcribe_audio(self, command: TranscribeAudioCommand) -> Dict[str, Any]:
        """
        Handle a TranscribeAudioCommand by processing audio data.
        
        Args:
            command: The TranscribeAudioCommand
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        self.logger.debug(f"Handling TranscribeAudioCommand for session {command.session_id}")
        
        try:
            # Get or create session
            session = self._get_session(command.session_id)
            if session is None:
                # Create a new session since one wasn't found
                start_command = StartTranscriptionSessionCommand(
                    session_id=command.session_id,
                    language=command.language
                )
                self._handle_start_session(start_command)
                session = self._get_session(command.session_id)
            
            # Add audio to session
            session.add_audio_chunk(command.audio_chunk)
            
            # Process audio with transcription engine
            result = self.transcription_manager.transcribe(
                command.audio_chunk,
                is_first_chunk=command.is_first_chunk,
                is_last_chunk=command.is_last_chunk,
                options={
                    'language': command.language or session.language,
                    'timestamp_ms': command.timestamp_ms,
                    **command.options
                }
            )
            
            # Handle errors
            if 'error' in result:
                self._handle_transcription_error(
                    session.session_id,
                    result['error'],
                    command.timestamp_ms
                )
                return {'error': result['error']}
            
            # Convert result to TranscriptionResult model
            transcription_result = TranscriptionResult(
                text=result.get('text', ''),
                is_final=result.get('is_final', False),
                session_id=session.session_id,
                timestamp=command.timestamp_ms,
                language=result.get('language', session.language),
                confidence=result.get('confidence', 1.0),
                processing_time=result.get('processing_time')
            )
            
            # Add result to session
            session.add_result(transcription_result)
            
            # Publish update event
            self._publish_transcription_update(
                session.session_id,
                transcription_result
            )
            
            # Close session if this is the final chunk
            if command.is_last_chunk and result.get('is_final', False):
                session.close()
            
            return transcription_result.to_dict()
            
        except Exception as e:
            error_message = f"Error transcribing audio: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            self._handle_transcription_error(
                command.session_id,
                error_message,
                command.timestamp_ms
            )
            return {'error': error_message}
    
    def _handle_configure_transcription(self, command: ConfigureTranscriptionCommand) -> bool:
        """
        Handle a ConfigureTranscriptionCommand by updating configuration.
        
        Args:
            command: The ConfigureTranscriptionCommand
            
        Returns:
            bool: True if configuration was successful
        """
        self.logger.info(f"Handling ConfigureTranscriptionCommand, engine_type={command.engine_type}")
        
        try:
            # Create config dict from command parameters
            config_dict = {
                'engine_type': command.engine_type,
                'model_name': command.model_name,
                'language': command.language,
                'compute_type': command.compute_type,
                'beam_size': command.beam_size,
                'streaming': command.streaming,
                'chunk_duration_ms': command.chunk_duration_ms,
                'chunk_overlap_ms': command.chunk_overlap_ms
            }
            
            # Add OpenAI API key if provided
            if hasattr(command, 'openai_api_key') and command.openai_api_key:
                config_dict['openai_api_key'] = command.openai_api_key
            
            # Add any additional options
            config_dict.update(command.options)
            
            # Update default config
            self.default_config = TranscriptionConfig(**config_dict)
            
            # If engine type is being changed and the manager is running, stop it
            # so that we can restart with the correct engine type
            if self.transcription_manager.is_running() and self.transcription_manager.engine_type != command.engine_type:
                self.logger.info(f"Engine type changing from {self.transcription_manager.engine_type} to {command.engine_type}, restarting manager")
                self.transcription_manager.stop()
            
            # If manager is running, configure it
            if self.transcription_manager.is_running():
                return self.transcription_manager.configure(config_dict)
            
            # If not running, we'll use these settings when it starts
            return True
            
        except Exception as e:
            error_message = f"Error configuring transcription: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return False
    
    def _handle_start_session(self, command: StartTranscriptionSessionCommand) -> Dict[str, Any]:
        """
        Handle a StartTranscriptionSessionCommand by creating a new session.
        
        Args:
            command: The StartTranscriptionSessionCommand
            
        Returns:
            Dict[str, Any]: Session information
        """
        self.logger.info(f"Handling StartTranscriptionSessionCommand, session_id={command.session_id}")
        
        try:
            # Check if session already exists
            session = self._get_session(command.session_id)
            if session is not None:
                self.logger.warning(f"Session {command.session_id} already exists")
                return {"session_id": session.session_id, "already_exists": True}
            
            # Create session config by merging default with command config
            config = TranscriptionConfig(
                **{**self.default_config.to_dict(), **command.config}
            )
            
            # Override streaming setting if specified
            if command.streaming is not None:
                config.streaming = command.streaming
            
            # Override language if specified
            if command.language is not None:
                config.language = command.language
            
            # Create new session
            session = TranscriptionSession(
                session_id=command.session_id,
                config=config,
                language=config.language
            )
            
            # Store session and track activity
            self.sessions[session.session_id] = session
            self._update_session_activity(session.session_id)
            
            # Ensure transcription manager is running with correct configuration
            if not self.transcription_manager.is_running():
                # Add configuration for quick mode based on streaming setting
                config_dict = config.to_dict()
                config_dict['quick_mode'] = not config.streaming  # Use parallel/quick mode when not streaming
                
                success = self.transcription_manager.start(
                    engine_type=config.engine_type,
                    engine_config=config_dict
                )
                if not success:
                    raise RuntimeError("Failed to start transcription engine")
            else:
                # Configure transcription manager with session settings
                config_dict = config.to_dict()
                config_dict['quick_mode'] = not config.streaming
                self.transcription_manager.configure(config_dict)
            
            # Publish event
            self._publish_transcription_started(
                session.session_id,
                session.language
            )
            
            return {"session_id": session.session_id, "created": True}
            
        except Exception as e:
            error_message = f"Error starting transcription session: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            self._handle_transcription_error(
                command.session_id,
                error_message,
                0.0
            )
            return {"error": error_message}
    
    def _handle_stop_session(self, command: StopTranscriptionSessionCommand) -> Dict[str, Any]:
        """
        Handle a StopTranscriptionSessionCommand by closing a session.
        
        Args:
            command: The StopTranscriptionSessionCommand
            
        Returns:
            Dict[str, Any]: Operation result
        """
        self.logger.info(f"Handling StopTranscriptionSessionCommand, session_id={command.session_id}")
        
        try:
            # Get session
            session = self._get_session(command.session_id)
            if session is None:
                self.logger.warning(f"Session {command.session_id} not found")
                return {"session_id": command.session_id, "found": False}
            
            # Process remaining audio if requested
            if command.flush_remaining_audio and session.audio_sample_count > 0:
                # Get combined audio
                audio_data = session.get_combined_audio()
                
                # Process as last chunk
                result = self.transcription_manager.transcribe(
                    audio_data,
                    is_first_chunk=False,
                    is_last_chunk=True,
                    options={
                        'language': session.language
                    }
                )
                
                # Convert result to TranscriptionResult model if successful
                if 'error' not in result:
                    transcription_result = TranscriptionResult(
                        text=result.get('text', ''),
                        is_final=True,
                        session_id=session.session_id,
                        timestamp=time.time() * 1000,  # Current time in ms
                        language=result.get('language', session.language),
                        confidence=result.get('confidence', 1.0),
                        processing_time=result.get('processing_time')
                    )
                    
                    # Add final result to session
                    session.add_result(transcription_result)
                    
                    # Publish update event
                    self._publish_transcription_update(
                        session.session_id,
                        transcription_result
                    )
            
            # Save results if requested
            if command.save_results and command.output_path:
                self._save_results(session, command.output_path)
            
            # Clean up session using our helper method
            self._cleanup_session(command.session_id)
            
            # Stop transcription manager if no active sessions
            if not self.sessions and self.transcription_manager.is_running():
                self.transcription_manager.stop()
            
            return {"session_id": command.session_id, "closed": True}
            
        except Exception as e:
            error_message = f"Error stopping transcription session: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"error": error_message}
    
    def _get_session(self, session_id: str) -> Optional[TranscriptionSession]:
        """
        Get a transcription session by ID.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Optional[TranscriptionSession]: The session or None if not found
        """
        session = self.sessions.get(session_id)
        
        # Update activity tracking if session exists
        if session is not None:
            self._update_session_activity(session_id)
            
        # Check for expired sessions periodically
        self._check_expired_sessions()
            
        return session
    
    def _save_results(self, session: TranscriptionSession, output_path: str) -> None:
        """
        Save session results to a file.
        
        Args:
            session: The session to save results for
            output_path: Path to save the results to
        """
        try:
            import json
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare results
            results = {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.last_activity_time,
                "duration_ms": session.duration_ms,
                "language": session.language,
                "transcription_text": session.current_text,
                "results": [result.to_dict() for result in session.results]
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Saved transcription results to {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving results to {output_path}: {str(e)}")
    
    def _publish_transcription_started(self, session_id: str, language: Optional[str]) -> None:
        """
        Publish a TranscriptionStartedEvent.
        
        Args:
            session_id: Session ID
            language: Language code
        """
        event = TranscriptionStartedEvent(
            session_id=session_id,
            language=language,
            audio_timestamp=time.time() * 1000  # Current time in ms
        )
        self.event_bus.publish(event)
    
    def _publish_transcription_update(self, session_id: str, result: TranscriptionResult) -> None:
        """
        Publish a TranscriptionUpdatedEvent.
        
        Args:
            session_id: Session ID
            result: Transcription result
        """
        event = TranscriptionUpdatedEvent(
            session_id=session_id,
            text=result.text,
            is_final=result.is_final,
            confidence=result.confidence,
            language=result.language,
            audio_timestamp=result.timestamp,
            processing_time=result.processing_time,
            segments=[segment.__dict__ for segment in result.segments],
            metadata=result.metadata
        )
        self.event_bus.publish(event)
    
    def _handle_transcription_error(self, session_id: str, error_message: str, audio_timestamp: float) -> None:
        """
        Handle and publish a transcription error.
        
        Args:
            session_id: Session ID
            error_message: Error message
            audio_timestamp: Audio timestamp in milliseconds
        """
        self.logger.error(f"Transcription error in session {session_id}: {error_message}")
        
        # Create error event
        event = TranscriptionErrorEvent(
            session_id=session_id,
            error_message=error_message,
            error_type="TranscriptionError",
            audio_timestamp=audio_timestamp
        )
        
        # Publish event
        self.event_bus.publish(event)
    
    def _check_expired_sessions(self, force: bool = False) -> None:
        """
        Check for and clean up expired sessions.
        
        This method is called periodically to remove inactive sessions
        and prevent memory leaks. Sessions are considered expired if they
        have had no activity for session_expiry_seconds.
        
        Args:
            force: If True, force a cleanup regardless of counter
        """
        # Only check periodically unless forced
        if not force:
            self._cleanup_counter += 1
            if self._cleanup_counter < self._cleanup_interval:
                return
            
        # Reset counter
        self._cleanup_counter = 0
        
        # Get current time
        current_time = time.time()
        expired_sessions = []
        
        # Find expired sessions
        for session_id, last_activity in list(self.session_last_activity.items()):
            if current_time - last_activity > self.session_expiry_seconds:
                expired_sessions.append(session_id)
                
        # Clean up expired sessions
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
            
        # Log if sessions were cleaned up
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _cleanup_session(self, session_id: str) -> None:
        """
        Clean up a specific session and its resources.
        
        Args:
            session_id: ID of the session to clean up
        """
        if session_id in self.sessions:
            try:
                # Get session
                session = self.sessions[session_id]
                
                # Close session properly
                session.close()
                
                # Remove from the sessions dictionary
                del self.sessions[session_id]
                
                # Remove from last activity tracking
                if session_id in self.session_last_activity:
                    del self.session_last_activity[session_id]
                    
                self.logger.debug(f"Cleaned up session: {session_id}")
            except Exception as e:
                self.logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def _update_session_activity(self, session_id: str) -> None:
        """
        Update the last activity timestamp for a session.
        
        Args:
            session_id: ID of the session to update
        """
        self.session_last_activity[session_id] = time.time()
        
        # Check if we need to enforce the maximum sessions limit
        if len(self.sessions) > self.max_sessions:
            # Find oldest session by activity time
            oldest_session_id = min(self.session_last_activity.items(), key=lambda x: x[1])[0]
            self.logger.warning(f"Session limit reached, cleaning up oldest session: {oldest_session_id}")
            self._cleanup_session(oldest_session_id)
            
        # Periodically check for expired sessions
        self._check_expired_sessions()
    
    def cleanup(self) -> None:
        """Clean up all resources used by the handler."""
        self.logger.info("Cleaning up TranscriptionCommandHandler")
        
        # Force cleanup of all expired sessions
        self._check_expired_sessions(force=True)
        
        # Stop transcription manager
        if self.transcription_manager.is_running():
            self.transcription_manager.stop()
        
        # Close all sessions
        for session_id, session in list(self.sessions.items()):
            try:
                session.close()
            except Exception as e:
                self.logger.error(f"Error closing session {session_id}: {e}")
        
        # Clear session dictionaries
        self.sessions.clear()
        self.session_last_activity.clear()

    # Additional method to handle VAD events
    def on_silence_detected(self, session_id: str, audio_reference: Any, duration: float) -> None:
        """
        Handle silence detection events for transcription.
        
        Args:
            session_id: Session ID from the VAD module
            audio_reference: Audio data from the speech segment
            duration: Duration of the speech segment in seconds
        """
        self.logger.info(f"Received silence detection event, speech duration: {duration:.2f}s")
        
        # Check if audio reference is empty or None
        # Handle NumPy arrays specifically to avoid ambiguous truth value error
        if audio_reference is None or (hasattr(audio_reference, 'size') and audio_reference.size == 0):
            self.logger.warning("Empty audio reference received, skipping transcription")
            return
        
        # Skip extremely short speech segments (likely false positives)
        # The 0.2 seconds threshold can be adjusted based on your needs
        if duration < 0.2:
            self.logger.warning(f"Speech segment too short ({duration:.2f}s < 0.2s), skipping transcription")
            return
            
        # Generate a session ID if none provided
        if not session_id:
            session_id = str(uuid.uuid4())
            
        # Save audio to a temporary WAV file
        # This helps the transcription engine to handle the audio reliably
        import tempfile
        import numpy as np
        import soundfile as sf
        import os
        
        # Create a unique filename for this segment
        audio_id = session_id.split('-')[-1][:8]  # Use the last part of session ID
        temp_dir = tempfile.gettempdir()
        wav_path = os.path.join(temp_dir, f"speech_{audio_id}.wav")
        
        # Make sure audio is the right format
        if isinstance(audio_reference, np.ndarray):
            # Log audio statistics for debugging
            if audio_reference.size > 0:
                self.logger.info(f"Audio stats: shape={audio_reference.shape}, "
                               f"duration={len(audio_reference)/16000:.2f}s, "
                               f"min={np.min(audio_reference):.4f}, max={np.max(audio_reference):.4f}, "
                               f"mean={np.mean(audio_reference):.4f}, rms={np.sqrt(np.mean(np.square(audio_reference))):.4f}")
                
                # Log audio RMS for informational purposes only
                rms = np.sqrt(np.mean(np.square(audio_reference)))
                self.logger.info(f"Audio RMS energy level: {rms:.4f}")
            
            # TEMPORARY DEBUG: Save audio to debug directory for debugging
            try:
                import os.path
                # Ensure we're saving to the debug directory, not project root
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
                debug_dir = os.path.join(base_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Use timestamp in filename to avoid overwriting previous files
                timestamp = int(time.time())
                debug_path = os.path.join(debug_dir, f"speech_{timestamp}.wav")
                sf.write(debug_path, audio_reference, 16000, format='WAV', subtype='PCM_16')
                self.logger.info(f"DEBUGGING: Saved audio to debug dir: {debug_path}")
            except Exception as e:
                self.logger.error(f"Error saving debug audio file: {e}")
            
            # Save the audio data to a WAV file
            try:
                sf.write(wav_path, audio_reference, 16000, format='WAV', subtype='PCM_16')
                self.logger.info(f"Saved audio to temporary file: {wav_path}")
            except Exception as e:
                self.logger.error(f"Error saving audio to temporary file: {e}")
                return
            
            # Process the complete speech segment with configured language
            command = TranscribeAudioCommand(
                audio_chunk=wav_path,  # Pass the file path instead of the raw audio
                session_id=session_id,
                is_first_chunk=True,
                is_last_chunk=True,
                timestamp_ms=time.time() * 1000,
                language=self.default_config.language  # Pass the configured language
            )
        else:
            # Use the original audio reference if it's not a numpy array
            # (though this should not happen with our current implementation)
            self.logger.warning(f"Non-numpy audio reference received (type: {type(audio_reference)})")
            command = TranscribeAudioCommand(
                audio_chunk=audio_reference,
                session_id=session_id,
                is_first_chunk=True,
                is_last_chunk=True,
                timestamp_ms=time.time() * 1000,
                language=self.default_config.language  # Pass the configured language
            )
        
        self.logger.info(f"Processing speech segment (duration: {duration:.2f}s) with session: {session_id}, language: {self.default_config.language}")
        self.handle(command)