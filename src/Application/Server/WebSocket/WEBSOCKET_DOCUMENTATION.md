# WebSocket Architecture Documentation

## Overview

The WebSocket module in Realtime_mlx_STT provides real-time bidirectional communication between the server and clients. It enables live streaming of transcription results, wake word detections, and other events to connected clients.

## Architecture

### Core Components

#### WebSocketManager (`WebSocketManager.py`)
The central manager for all WebSocket connections, handling:
- Client connection/disconnection lifecycle
- Message broadcasting to all connected clients
- Thread-safe event broadcasting from any thread
- Error handling and automatic cleanup of disconnected clients

### Connection Management

#### Client Registration
- Clients connect to the `/events` endpoint
- Each connection is tracked in `active_connections` set
- Automatic cleanup on disconnect or error

```python
# Connection flow
1. Client connects to ws://localhost:8080/events
2. Server accepts connection via FastAPI WebSocket
3. WebSocketManager.register() adds to active connections
4. Event loop is set for thread-safe broadcasting
```

#### Disconnection Handling
- Graceful disconnect via `WebSocketDisconnect` exception
- Error-based disconnect with automatic unregistration
- Cleanup of failed connections during broadcast

### Message Protocol

#### Outbound Messages (Server → Client)
All messages follow a standard JSON format:

```json
{
  "event": "event_type",
  "field1": "value1",
  "field2": "value2"
}
```

#### Event Types

1. **Transcription Events**
```json
{
  "event": "transcription",
  "text": "transcribed text",
  "is_final": true,
  "session_id": "uuid-string"
}
```

2. **Wake Word Events**
```json
{
  "event": "wake_word",
  "word": "detected_word",
  "confidence": 0.95,
  "timestamp": 1234567890
}
```

#### Inbound Messages (Client → Server)
Currently, the server accepts JSON messages but doesn't process specific commands through WebSocket. Future expansion planned for:
- Session control commands
- Configuration updates
- Real-time audio streaming

### Broadcasting Patterns

#### Thread-Safe Broadcasting
The WebSocketManager handles broadcasts from multiple threads:

1. **From Async Context** (same event loop):
   - Direct task creation using `current_loop.create_task()`
   - No thread synchronization needed

2. **From Sync Context** (different thread):
   - Uses `asyncio.run_coroutine_threadsafe()`
   - Requires event loop reference set via `set_event_loop()`

```python
# Example: Broadcasting from event handler (sync thread)
def handle_transcription_update(self, event):
    self.websocket_manager.broadcast_event("transcription", {
        "text": event.text,
        "is_final": event.is_final,
        "session_id": event.session_id
    })
```

### Integration with EventBus

The WebSocket system integrates seamlessly with the application's EventBus:

1. **Event Subscription**: Server subscribes to relevant events
   - `TranscriptionUpdatedEvent`
   - `WakeWordDetectedEvent`
   - Additional events can be added

2. **Event Handlers**: Convert internal events to WebSocket messages
   - Extract relevant data from events
   - Format as JSON message
   - Broadcast to all connected clients

3. **Decoupling**: WebSocket layer doesn't know about internal architecture
   - Events are translated at the boundary
   - Clean separation of concerns

## Security Considerations

### Current Implementation

1. **CORS Configuration**
   - Configurable allowed origins (default: "*")
   - Full CORS middleware on FastAPI app

2. **Authentication (Planned)**
   - Config supports `auth_enabled` and `auth_token`
   - Not yet implemented in WebSocket connections
   - Future: Token-based auth on connection

3. **Connection Limits**
   - No current limit on concurrent connections
   - Each connection tracked in memory
   - Consider implementing connection pooling for production

### Security Best Practices

1. **Input Validation**
   - All incoming messages should be validated
   - Prevent injection attacks
   - Limit message size

2. **Rate Limiting**
   - Consider implementing message rate limits
   - Prevent DoS attacks
   - Track per-client metrics

3. **TLS/SSL**
   - Use wss:// in production
   - Encrypt all data in transit

## Performance Considerations

### Current Architecture

1. **Connection Handling**
   - Set-based storage: O(1) add/remove
   - Linear broadcast: O(n) for n clients
   - Async I/O for non-blocking operations

2. **Message Broadcasting**
   - Sequential sending to all clients
   - Failed connections removed during broadcast
   - No message queuing or buffering

3. **Thread Safety**
   - Event loop integration for cross-thread calls
   - No explicit locking needed (GIL + async)

### Scaling Limits

1. **Single Server Instance**
   - Limited by Python's single process
   - Practical limit: ~1000 concurrent connections
   - Memory usage: ~1MB per connection

2. **Broadcast Performance**
   - Linear degradation with client count
   - Consider pub/sub for large scale
   - No built-in message compression

### Optimization Opportunities

1. **Message Batching**
   - Combine multiple events in time window
   - Reduce network overhead
   - Implement client-side buffering

2. **Selective Broadcasting**
   - Room/channel concept for grouped clients
   - Event filtering based on client preferences
   - Reduce unnecessary traffic

3. **Connection Pooling**
   - Reuse connections where possible
   - Implement connection limits
   - Add connection health checks

## Usage Examples

### Server-Side Setup
```python
# In ServerModule.__init__
self.websocket_manager = WebSocketManager()

# Register WebSocket endpoint
@self.app.websocket("/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    self.websocket_manager.register(websocket)
    # ... handle connection
```

### Client-Side Connection (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8080/events');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.event === 'transcription') {
        console.log('Transcription:', data.text);
    } else if (data.event === 'wake_word') {
        console.log('Wake word detected:', data.word);
    }
};
```

### Broadcasting Events
```python
# From any thread
websocket_manager.broadcast_event("custom_event", {
    "message": "Hello, clients!",
    "timestamp": time.time()
})
```

## Error Handling

### Connection Errors
- Automatic unregistration on send failure
- Logged errors with client count updates
- No exception propagation to callers

### Broadcast Errors
- Failed clients removed from active set
- Broadcast continues to remaining clients
- Errors logged but not fatal

### Event Loop Errors
- Fallback to logging if no event loop
- Graceful degradation without crashes
- Clear error messages for debugging

## Future Enhancements

1. **Bidirectional Commands**
   - Process client commands via WebSocket
   - Real-time configuration updates
   - Session control messages

2. **Audio Streaming**
   - Binary WebSocket frames for audio
   - Chunk-based streaming protocol
   - Backpressure handling

3. **Authentication & Authorization**
   - JWT token validation
   - Per-client permissions
   - Secure session management

4. **Scaling Solutions**
   - Redis pub/sub for multi-server
   - WebSocket connection routing
   - Load balancing strategies

5. **Monitoring & Metrics**
   - Connection statistics
   - Message throughput tracking
   - Client behavior analytics