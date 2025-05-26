# Orchestration Unification Plan

## Problem Statement

The codebase currently has two parallel orchestration implementations:
1. **API Layer** (`realtime_mlx_stt/session.py`): Complete pipeline orchestration for Python API
2. **Application Layer** (`SystemController` + `TranscriptionController`): Same orchestration for REST/WebSocket API

This duplication leads to:
- Maintenance burden (bugs must be fixed in two places)
- Risk of divergent behavior
- Duplicated testing requirements
- Confusion about which layer owns orchestration logic

## Current Architecture Analysis

### Duplicated Orchestration Logic
Both layers implement identical:
- Module initialization sequences (Transcription → VAD → Audio → Wake Word)
- State management (session tracking, active features)
- Configuration handling (profiles, custom overrides)
- Event subscription patterns
- Cleanup sequences (reverse order shutdown)

### Key Differences
- **API Layer**: Direct Python callbacks, single session, synchronous
- **Application Layer**: REST/WebSocket events, multi-session, async-ready

## Proposed Solution

### Option 1: API Uses Application Orchestration (Recommended)

Make the API layer a thin wrapper around the Application layer's orchestration:

```
┌─────────────────────────────────────┐
│            API Layer                 │
│   (Thin wrapper for Python users)    │
├─────────────────────────────────────┤
│        Application Layer             │
│  (SystemController orchestration)    │
├─────────────────────────────────────┤
│         Features Layer               │
│    (Business logic modules)          │
└─────────────────────────────────────┘
```

#### Implementation Steps:

1. **Refactor SystemController**
   - Extract orchestration logic into reusable methods
   - Separate HTTP-specific code from core orchestration
   - Create a `SystemOrchestrator` class

2. **Update API Layer**
   - Remove orchestration logic from `session.py`
   - Use `SystemOrchestrator` instead
   - Convert callbacks to event subscriptions
   - Maintain backward compatibility

3. **Benefits**
   - Single source of truth for orchestration
   - Consistent behavior between APIs
   - Easier maintenance
   - Reduced code duplication

### Option 2: Extract Common Orchestration Service

Create a new shared service layer:

```
┌─────────────────────┐  ┌─────────────────────┐
│     API Layer       │  │  Application Layer   │
└──────────┬──────────┘  └──────────┬──────────┘
           │                         │
           v                         v
┌─────────────────────────────────────┐
│      Orchestration Service          │
│  (Shared session management)        │
├─────────────────────────────────────┤
│         Features Layer              │
└─────────────────────────────────────┘
```

#### Implementation Steps:

1. **Create OrchestrationService**
   - Extract common logic from both layers
   - Define interface for session management
   - Handle module coordination

2. **Update Both Layers**
   - Remove duplicated orchestration
   - Use OrchestrationService
   - Adapt to specific needs (callbacks vs events)

3. **Benefits**
   - Clean separation of concerns
   - Both layers remain thin
   - Flexibility for different use cases

## Recommendation: Option 1

**Rationale:**
- SystemController already has mature, tested orchestration logic
- Minimal changes required
- Natural hierarchy: API wraps Application
- Follows common patterns (e.g., SDK wrapping REST API)

## Implementation Plan

### Phase 1: Refactor SystemController (1-2 days)
1. Extract orchestration methods:
   - `initialize_pipeline(config)`
   - `start_pipeline()`
   - `stop_pipeline()`
   - `configure_profile(profile_name, custom_config)`

2. Separate concerns:
   - Move HTTP handling to controller
   - Move orchestration to service class
   - Keep REST endpoints minimal

### Phase 2: Update API Layer (1-2 days)
1. Refactor `TranscriptionSession`:
   ```python
   class TranscriptionSession:
       def __init__(self, ...):
           self._orchestrator = SystemOrchestrator()
           self._setup_callbacks()
       
       def start(self):
           # Use orchestrator instead of direct commands
           config = self._build_config()
           return self._orchestrator.start_pipeline(config)
   ```

2. Maintain API compatibility:
   - Keep existing method signatures
   - Convert events to callbacks
   - Preserve behavior

### Phase 3: Testing & Migration (1 day)
1. Ensure all existing tests pass
2. Add integration tests for unified orchestration
3. Update documentation
4. Migration guide for any breaking changes

## Success Criteria

1. **Zero Duplication**: Single orchestration implementation
2. **Backward Compatibility**: Existing API users unaffected
3. **Consistent Behavior**: Both APIs behave identically
4. **Maintainability**: Changes in one place affect both APIs
5. **Test Coverage**: Comprehensive tests for orchestration

## Risks & Mitigation

1. **Risk**: Breaking existing API users
   - **Mitigation**: Extensive testing, compatibility layer

2. **Risk**: Performance overhead from additional layer
   - **Mitigation**: Minimal wrapper, no significant overhead

3. **Risk**: Complexity increase
   - **Mitigation**: Clear documentation, clean interfaces

## Alternative: Keep Current Architecture

If refactoring is deemed too risky:
1. Document the intentional duplication
2. Create shared test suites
3. Establish process for synchronized updates
4. Consider this refactoring for v2.0

## Decision Required

Choose between:
- **Option 1**: API uses Application orchestration (recommended)
- **Option 2**: Extract common orchestration service
- **Option 3**: Keep current architecture with better documentation

Timeline: 3-5 days for implementation and testing