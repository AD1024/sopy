# Error Recovery: Building Resilient Agent Workflows

One of SOPy's key strengths is its built-in error recovery mechanisms. This tutorial explains how SOPy handles failures and how to design robust recovery strategies for your agent workflows.

## Why Error Recovery Matters

LLM agents operate in unpredictable environments where failures are inevitable:
- **Tool failures**: APIs go down, files are locked, network issues occur
- **Logic errors**: Unexpected conditions, invalid inputs, assertion failures
- **Agent confusion**: LLM makes incorrect decisions or gets stuck
- **Resource constraints**: Memory limits, timeouts, rate limiting

Without proper error recovery, agents fail catastrophically. SOPy provides multiple layers of protection to ensure graceful degradation and automatic recovery.

## SOPy's Error Recovery Architecture

SOPy implements a multi-layered error recovery system:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│              Custom error handlers, validation              │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    SOP Recovery Layer                       │
│              Retry mechanisms, recovery prompts             │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Procedure Layer                           │
│    Handler error returns, assertions, validation logic      │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Event Layer                             │
│        Event validation, payload checking, type safety      │
└─────────────────────────────────────────────────────────────┘
```

## Handler-Level Error Recovery

The first line of defense is in your event handlers. Handlers can return error messages to trigger recovery:

### 1. Simple Error Messages

```python
@handler
def handle_file_operation(self, state: MyState, event: FileOperationResult):
    if not event.payload.success:
        return f"File operation failed: {event.payload.error_message}. Please try again."
    
    # Success case
    state.processed_files.append(event.payload.filename)
    return NextProcedure()
```

### 2. Conditional Error Recovery

```python
@handler
def handle_api_response(self, state: APIState, event: APIResponse):
    if event.payload.status_code == 429:  # Rate limited
        return "API rate limit exceeded. Please wait and try again."
    elif event.payload.status_code >= 500:  # Server error
        return f"Server error ({event.payload.status_code}). Please retry the request."
    elif event.payload.status_code >= 400:  # Client error
        return f"Request error: {event.payload.error_message}. Please check the request and try again."
    
    # Success case
    state.api_data = event.payload.data
    return ProcessData()
```

### 3. Assertion-Based Recovery

SOPy automatically converts assertion failures into recovery prompts:

```python
@handler
def handle_data_validation(self, state: DataState, event: DataReceived):
    data = event.payload
    
    # These assertions become recovery prompts automatically
    assert data is not None, "No data received. Please ensure data source is available."
    assert len(data) > 0, "Empty data set received. Please provide valid data."
    assert all(item.get('id') for item in data), "Data items missing required 'id' field. Please check data format."
    
    # Process valid data
    state.validated_data = data
    return ProcessValidatedData()
```

## SOP-Level Error Recovery

The SOP class provides higher-level recovery mechanisms through the `with_recovery()` method:

### Basic Recovery Setup

```python
def create_resilient_sop():
    sop = SOP(
        state=MyState(),
        init_proc=StartProcedure()
    ).observes(MyEvent)
    
    # Add recovery mechanism
    def error_handler(prompt):
        print(f"Error occurred: {prompt}")
        # Could call LLM agent, log error, send notification, etc.
        return "I encountered an error and will try to recover."
    
    return sop.with_recovery(
        error_handler=error_handler,
        retry_limits=3
    )
```

### Advanced Recovery with Agent Integration

```python
from sopy.interface.strands import StrandsAgent

def create_agent_with_recovery():
    # Create the SOP
    sop = SOP(state=TaskState(), init_proc=StartTask())
    
    # Create agent for recovery
    recovery_agent = StrandsAgent(
        tools=[diagnostic_tool, repair_tool],
        system_prompt="You are a recovery agent. Analyze errors and take corrective action."
    )
    
    # Recovery handler that uses the agent
    def intelligent_recovery(error_prompt):
        recovery_response = recovery_agent(f"""
        An error occurred in the workflow: {error_prompt}
        
        Please analyze the error and take appropriate corrective action.
        Available tools: diagnostic_tool, repair_tool
        """)
        return recovery_response
    
    return sop.with_recovery(
        error_handler=intelligent_recovery,
        retry_limits=5
    )
```

## Recovery Patterns

### 1. Retry with Backoff

```python
class RetryProcedure(Procedure[MyState]):
    def __init__(self):
        self.attempt_count = 0
        self.max_attempts = 3
        self.backoff_seconds = [1, 2, 4]  # Exponential backoff
    
    @handler
    def handle_operation_failed(self, state: MyState, event: OperationFailed):
        self.attempt_count += 1
        
        if self.attempt_count >= self.max_attempts:
            return "Maximum retry attempts exceeded. Operation failed permanently."
        
        backoff_time = self.backoff_seconds[min(self.attempt_count - 1, len(self.backoff_seconds) - 1)]
        return f"Operation failed (attempt {self.attempt_count}/{self.max_attempts}). Waiting {backoff_time} seconds before retry."
```

### 2. Fallback Strategies

```python
class FallbackProcedure(Procedure[MyState]):
    @handler
    def handle_primary_method_failed(self, state: MyState, event: PrimaryMethodFailed):
        if not state.tried_fallback:
            state.tried_fallback = True
            return TryFallbackMethod()
        else:
            return "Both primary and fallback methods failed. Manual intervention required."
    
    @handler
    def handle_fallback_method_failed(self, state: MyState, event: FallbackMethodFailed):
        return "All recovery methods exhausted. Escalating to manual resolution."
```

### 3. State Rollback

```python
class TransactionalProcedure(Procedure[MyState]):
    def __init__(self):
        self.checkpoint = None
    
    @handler
    def handle_begin_transaction(self, state: MyState, event: BeginTransaction):
        # Create checkpoint
        self.checkpoint = deepcopy(state)
        return ProcessTransaction()
    
    @handler
    def handle_transaction_failed(self, state: MyState, event: TransactionFailed):
        if self.checkpoint:
            # Rollback to checkpoint
            for key, value in self.checkpoint.__dict__.items():
                setattr(state, key, value)
            return "Transaction failed. State rolled back to checkpoint."
        else:
            return "Transaction failed and no checkpoint available. Manual recovery required."
```

### 4. Circuit Breaker Pattern

```python
class CircuitBreakerProcedure(Procedure[MyState]):
    def __init__(self):
        self.failure_count = 0
        self.failure_threshold = 5
        self.circuit_open = False
        self.last_failure_time = None
        self.recovery_timeout = 60  # seconds
    
    @handler
    def handle_operation_request(self, state: MyState, event: OperationRequest):
        if self.circuit_open:
            # Check if we should try to close the circuit
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.circuit_open = False
                self.failure_count = 0
            else:
                return "Circuit breaker is open. Service temporarily unavailable."
        
        return ProcessOperation()
    
    @handler
    def handle_operation_failed(self, state: MyState, event: OperationFailed):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.circuit_open = True
            return "Too many failures. Circuit breaker opened to prevent cascade failures."
        
        return "Operation failed. Will retry."
```

## Error Recovery in the File System Example

The file system operator demonstrates several recovery patterns:

### 1. Retry Logic in Copy Operations

```python
class Copy(Procedure[FSInfo]):
    def __init__(self, prompt: Prompt):
        self.retry = 2  # Retry attempts remaining

    @handler
    def handle_eCopyRequest(self, sigma: FSInfo, event: eCopyRequest):
        if cond(self.retry == 0, "retry limit is reached"):
            return Abort()  # Transition to cleanup procedure
        # ... validation logic
        return self

class BackUpCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:  # Copy failed
            self.retry -= 1
            return self  # Stay in same state, will retry
        # ... success logic
```

### 2. Comprehensive Rollback in Abort Procedure

```python
class Abort(Procedure[FSInfo]):
    """Handles complete rollback when operations fail"""
    
    @handler
    def handle_eFileDeleted(self, sigma: FSInfo, event: eFileDeleted):
        # Track cleanup progress
        if dst in sigma.files_in_backup:
            self.deleted_backup.add(dst)
        elif dst in sigma.files_in_migration:
            self.deleted_migration.add(dst)
        
        # Only complete when ALL created files are cleaned up
        if cond(self.deleted_backup == sigma.files_in_backup and 
                self.deleted_migration == sigma.files_in_migration,
                "backups and partially migrated files are deleted"):
            return End()
        
        return self
```

### 3. SOP-Level Recovery Integration

```python
def start_task(src_path: str, dst_path: str, backup_path: str):
    sop = FileSystemOperator(src_path, dst_path, backup_path)
    
    # Agent-based error recovery
    agent = StrandsAgent(tools=[read_dir, cp, delete], system_prompt=prompt)
    
    sop = sop.with_recovery(
        error_handler=lambda prompt: agent(f"Error occurred: {prompt}"),
        retry_limits=3
    )
    
    with sop:
        agent(f"Move all files from {src_path} to {dst_path} with backup at {backup_path}")
```

## Best Practices for Error Recovery

### 1. Design for Failure

Assume failures will happen and design your procedures accordingly:

```python
# Good - Explicit failure handling
@handler
def handle_api_call(self, state: MyState, event: APIResult):
    if event.payload.success:
        return ProcessSuccess()
    elif event.payload.retryable:
        return "API call failed but is retryable. Please try again."
    else:
        return FallbackProcedure()

# Avoid - Only handling success case
@handler
def handle_api_call(self, state: MyState, event: APIResult):
    return ProcessSuccess()  # What if it failed?
```

### 2. Provide Meaningful Error Messages

```python
# Good - Specific, actionable error messages
@handler
def handle_database_error(self, state: MyState, event: DatabaseError):
    error_code = event.payload.error_code
    if error_code == "CONNECTION_TIMEOUT":
        return "Database connection timed out. Please check network connectivity and retry."
    elif error_code == "PERMISSION_DENIED":
        return "Database access denied. Please verify credentials and permissions."
    else:
        return f"Database error ({error_code}): {event.payload.message}. Please check database status."

# Avoid - Generic error messages
@handler
def handle_database_error(self, state: MyState, event: DatabaseError):
    return "Database error occurred. Please try again."
```

### 3. Implement Graceful Degradation

```python
@handler
def handle_service_unavailable(self, state: MyState, event: ServiceUnavailable):
    if state.has_cached_data:
        state.using_cached_data = True
        return ProcessWithCachedData()
    elif state.has_fallback_service:
        return TryFallbackService()
    else:
        return "Primary service unavailable and no fallback available. Please try again later."
```

### 4. Log Recovery Actions

```python
from sopy.utils import Log

@handler
def handle_recovery_action(self, state: MyState, event: RecoveryNeeded):
    Log.warning(f"Recovery triggered: {event.payload.reason}")
    
    if self.can_auto_recover():
        Log.info("Attempting automatic recovery")
        return AutoRecoveryProcedure()
    else:
        Log.error("Manual intervention required")
        return "Automatic recovery not possible. Manual intervention required."
```

### 5. Test Recovery Scenarios

```python
def test_error_recovery():
    """Test that errors trigger proper recovery"""
    sop = create_test_sop()
    
    with sop:
        # Simulate failure
        EventBus.push_event(OperationFailed(payload="simulated failure"))
        
        # Verify recovery was triggered
        assert isinstance(sop.current_proc, RecoveryProcedure)
        
        # Verify state is consistent
        assert sop.state.error_count == 1
        assert sop.state.recovery_attempted == True
```

## Advanced Recovery Techniques

### 1. Multi-Level Recovery

```python
class MultiLevelRecovery(Procedure[MyState]):
    @handler
    def handle_operation_failed(self, state: MyState, event: OperationFailed):
        failure_type = event.payload.failure_type
        
        if failure_type == "TRANSIENT":
            return "Transient failure detected. Retrying immediately."
        elif failure_type == "RESOURCE":
            return WaitForResourcesProcedure()
        elif failure_type == "CONFIGURATION":
            return ReconfigureProcedure()
        else:
            return EscalateProcedure()
```

### 2. Recovery with External Systems

```python
@handler
def handle_critical_failure(self, state: MyState, event: CriticalFailure):
    # Notify external monitoring system
    monitoring_client.send_alert({
        "severity": "critical",
        "component": "sopy_workflow",
        "message": event.payload.error_message,
        "state": state.dict()
    })
    
    # Attempt recovery
    if state.recovery_attempts < 3:
        state.recovery_attempts += 1
        return RecoveryProcedure()
    else:
        return "Maximum recovery attempts exceeded. Escalating to operations team."
```

### 3. Predictive Recovery

```python
class PredictiveRecovery(Procedure[MyState]):
    @handler
    def handle_performance_degradation(self, state: MyState, event: PerformanceDegraded):
        # Proactively switch to more robust approach before failure occurs
        if event.payload.response_time > state.performance_threshold:
            Log.warning("Performance degradation detected. Switching to robust mode.")
            return RobustModeProcedure()
        
        return self
```

## Summary

SOPy's error recovery system provides multiple layers of protection:

- **Handler-level recovery**: Return error messages or use assertions
- **SOP-level recovery**: Retry mechanisms with custom error handlers
- **Pattern-based recovery**: Implement common patterns like retry, fallback, circuit breaker
- **State management**: Rollback and checkpoint capabilities
- **Integration**: Work with external monitoring and recovery systems

The key to effective error recovery is to:
1. **Anticipate failures** and design for them explicitly
2. **Provide clear error messages** that enable effective recovery
3. **Implement appropriate patterns** for different failure types
4. **Test recovery scenarios** to ensure they work correctly
5. **Monitor and log** recovery actions for observability

With proper error recovery design, your SOPy workflows can handle failures gracefully and continue operating reliably even in challenging environments.

## Next Steps

- [Tool Integration](07-tool-integration.md) - Learn about integrating with external tools
- [File System Example](08-file-system-example.md) - See error recovery in action
- [Advanced Patterns](09-advanced-patterns.md) - Complex recovery strategies
