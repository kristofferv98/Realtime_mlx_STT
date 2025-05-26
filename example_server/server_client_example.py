#!/usr/bin/env python3
"""
Server Client Example for Realtime_mlx_STT

This example demonstrates how to interact with the server API using both
REST endpoints and WebSocket for real-time transcription.

Prerequisites:
1. Start the server first:
   python examples/server_example.py

2. Install client dependencies:
   pip install requests websocket-client

Usage:
   python examples/server_client_example.py

"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Try to import required libraries
try:
    import requests
    import websocket
except ImportError:
    print("\nError: Client dependencies not installed!")
    print("Please install them with:")
    print("  pip install requests websocket-client")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerClient:
    """Client for interacting with the Realtime_mlx_STT server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/events"
        self.ws: Optional[websocket.WebSocket] = None
    
    def check_health(self) -> dict:
        """Check server health status."""
        try:
            response = requests.get(f"{self.base_url}/system/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check health: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_profiles(self) -> dict:
        """Get available transcription profiles."""
        try:
            response = requests.get(f"{self.base_url}/system/profiles")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get profiles: {e}")
            return {"error": str(e)}
    
    def start_system(self, profile: str = "vad-triggered", custom_config: Optional[dict] = None) -> dict:
        """Start system with specified profile and optional custom configuration."""
        try:
            payload = {
                "profile": profile
            }
            if custom_config:
                payload["custom_config"] = custom_config
                
            response = requests.post(
                f"{self.base_url}/system/start",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to start system: {e}")
            return {"error": str(e)}
    
    def stop_system(self) -> dict:
        """Stop the system."""
        try:
            response = requests.post(f"{self.base_url}/system/stop")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to stop transcription: {e}")
            return {"error": str(e)}
    
    def connect_websocket(self):
        """Connect to the WebSocket for real-time updates."""
        try:
            self.ws = websocket.create_connection(self.ws_url)
            logger.info(f"Connected to WebSocket: {self.ws_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    def disconnect_websocket(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.ws.close()
            self.ws = None
            logger.info("Disconnected from WebSocket")
    
    def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk via WebSocket."""
        if not self.ws:
            logger.error("WebSocket not connected")
            return
        
        try:
            # Send as binary message
            self.ws.send_binary(audio_data)
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
    
    def receive_transcription(self, timeout: float = 1.0) -> Optional[dict]:
        """Receive transcription update from WebSocket."""
        if not self.ws:
            logger.error("WebSocket not connected")
            return None
        
        try:
            self.ws.settimeout(timeout)
            message = self.ws.recv()
            return json.loads(message)
        except websocket.WebSocketTimeoutException:
            return None
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None


def demonstrate_rest_api(client: ServerClient):
    """Demonstrate REST API usage."""
    print("\n" + "="*60)
    print("REST API Demonstration")
    print("="*60)
    
    # Check health
    print("\n1. Checking server health...")
    health = client.check_health()
    print(f"   Health status: {json.dumps(health, indent=2)}")
    
    # Get profiles
    print("\n2. Getting available profiles...")
    profiles = client.get_profiles()
    print(f"   Available profiles: {json.dumps(profiles, indent=2)}")
    
    # Start system with custom config
    print("\n3. Starting system with custom configuration...")
    custom_config = {
        "transcription": {
            "engine": "mlx_whisper",
            "model": "whisper-large-v3-turbo",
            "language": "no"  # Norwegian
        },
        "vad": {
            "sensitivity": 0.7
        }
    }
    result = client.start_system(profile="vad-triggered", custom_config=custom_config)
    print(f"   Start result: {json.dumps(result, indent=2)}")
    
    # Wait a bit
    time.sleep(2)
    
    # Stop system
    print("\n4. Stopping system...")
    result = client.stop_system()
    print(f"   Stop result: {json.dumps(result, indent=2)}")


def demonstrate_websocket(client: ServerClient):
    """Demonstrate WebSocket usage."""
    print("\n" + "="*60)
    print("WebSocket Demonstration")
    print("="*60)
    
    # Connect to WebSocket
    print("\n1. Connecting to WebSocket...")
    if not client.connect_websocket():
        print("   Failed to connect!")
        return
    print("   Connected successfully!")
    
    # Simulate sending audio chunks
    print("\n2. Simulating audio streaming...")
    print("   (In a real application, you would send actual audio data)")
    
    # Send a few simulated chunks
    for i in range(3):
        # In reality, this would be actual audio data
        fake_audio = b"FAKE_AUDIO_DATA_" + str(i).encode()
        client.send_audio_chunk(fake_audio)
        print(f"   Sent chunk {i+1}")
        
        # Check for transcription updates
        update = client.receive_transcription(timeout=0.5)
        if update:
            print(f"   Received: {json.dumps(update, indent=2)}")
        
        time.sleep(1)
    
    # Disconnect
    print("\n3. Disconnecting...")
    client.disconnect_websocket()
    print("   Disconnected successfully!")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Realtime_mlx_STT Server Client Example")
    print("="*60)
    
    # Create client
    client = ServerClient()
    
    # Check if server is running
    print("\nChecking if server is running...")
    health = client.check_health()
    if "error" in health or health.get("status") != "healthy":
        print("\nError: Server is not running or not healthy!")
        print("Please start the server first with:")
        print("  python examples/server_example.py")
        sys.exit(1)
    
    print("Server is running and healthy!")
    
    try:
        # Demonstrate REST API
        demonstrate_rest_api(client)
        
        # Demonstrate WebSocket
        demonstrate_websocket(client)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Ensure WebSocket is closed
        client.disconnect_websocket()
    
    print("\nClient example completed!")


if __name__ == "__main__":
    main()