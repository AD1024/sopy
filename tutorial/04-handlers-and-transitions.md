# Handlers and Transitions: The Heart of SOPy Workflows

Event handlers are the core mechanism that drives state transitions in SOPy. They define how procedures respond to events and determine the next step in your workflow. This tutorial covers advanced handler patterns and transition strategies.

## Understanding Handlers

Handlers are methods decorated with `@handler` that process events and return the next action:

```python
from sopy import Procedure, handler, Event

class MyProcedure(Procedure[MyState]):
    @handler
    def handle_my_event(self, state: MyState, event: MyEvent) -> str | Procedure[MyState]:
        # Process the event
        # Return either:
        # - Next procedure (state transition)
        # - Error message string (triggers recovery)
        # - self (stay in current state)
        pass
```

## Handler Return Types

Handlers can return three types of values:

### 1. Next Procedure (State Transition)

```python
@handler
def handle_task_complete(self, state: TaskState, event: TaskComplete):
    if event.payload.success:
        state.completed_tasks.add(event.payload.task_id)
        return NextProcedure()  # Transition to next state
    else:
        return self  # Stay in current state
```

### 2. Error Message (Recovery Trigger)

```python
@handler
def handle_validation_result(self, state: DataState, event: ValidationResult):
    if not event.payload.valid:
        return f"Validation failed: {event.payload.error_message}. Please fix the data and try again."
    
    state.validated_data = event.payload.data
    return ProcessData()
```

### 3. Self Reference (Stay in State)

```python
@handler
def handle_partial_result(self, state: CollectionState, event: PartialResult):
    # Accumulate partial results
    state.partial_results.append(event.payload)
    
    # Check if we have all results
    if len(state.partial_results) >= state.expected_count:
        return ProcessResults()
    
    return self  # Stay in current state, wait for more results
```

## Advanced Handler Patterns

### 1. Conditional Transitions

Use complex logic to determine the next state:

```python
class ProcessingProcedure(Procedure[ProcessingState]):
    @handler
    def handle_processing_result(self, state: ProcessingState, event: ProcessingResult):
        result = event.payload
        
        if result.success:
            state.success_count += 1
            
            # Multiple exit conditions
            if state.success_count >= state.target_count:
                return CompleteProcedure()
            elif state.success_count >= state.batch_size:
                return ProcessNextBatch()
            else:
                return self  # Continue processing current batch
        
        else:
            state.error_count += 1
            
            # Error handling with different strategies
            if result.error_type == "retryable" and state.retry_count < state.max_retries:
                state.retry_count += 1
                return f"Retryable error occurred. Attempt {state.retry_count}/{state.max_retries}."
            elif result.error_type == "fatal":
                return AbortProcedure()
            else:
                return FallbackProcedure()
```

### 2. Dynamic Procedure Creation

Create procedures with context-specific configuration:

```python
@handler
def handle_task_assignment(self, state: WorkflowState, event: TaskAssignment):
    task = event.payload
    
    # Create procedure based on task type
    if task.type == "data_processing":
        return DataProcessingProcedure(
            prompt=make_prompt(f"Process {task.data_type} data using {task.algorithm}"),
            algorithm=task.algorithm,
            parameters=task.parameters
        )
    elif task.type == "file_operation":
        return FileOperationProcedure(
            prompt=make_prompt(f"Perform {task.operation} on {task.file_path}"),
            operation=task.operation,
            file_path=task.file_path
        )
    else:
        return f"Unknown task type: {task.type}. Please provide a valid task."
```

### 3. Multi-Event Handlers

Handle multiple related events in one procedure:

```python
class DataCollectionProcedure(Procedure[DataState]):
    def __init__(self):
        self.sources_completed = set()
        self.expected_sources = {"database", "api", "file"}
    
    @handler
    def handle_database_result(self, state: DataState, event: DatabaseResult):
        if event.payload.success:
            state.database_data = event.payload.data
            self.sources_completed.add("database")
        else:
            return f"Database query failed: {event.payload.error}"
        
        return self._check_completion(state)
    
    @handler
    def handle_api_result(self, state: DataState, event: APIResult):
        if event.payload.success:
            state.api_data = event.payload.data
            self.sources_completed.add("api")
        else:
            return f"API call failed: {event.payload.error}"
        
        return self._check_completion(state)
    
    @handler
    def handle_file_result(self, state: DataState, event: FileResult):
        if event.payload.success:
            state.file_data = event.payload.data
            self.sources_completed.add("file")
        else:
            return f"File read failed: {event.payload.error}"
        
        return self._check_completion(state)
    
    def _check_completion(self, state: DataState):
        """Check if all data sources are complete"""
        if self.sources_completed >= self.expected_sources:
            return MergeDataProcedure()
        return self
```

### 4. State-Dependent Transitions

Transition based on accumulated state:

```python
class AdaptiveProcedure(Procedure[AdaptiveState]):
    @handler
    def handle_performance_metric(self, state: AdaptiveState, event: PerformanceMetric):
        # Update performance history
        state.performance_history.append(event.payload.value)
        
        # Keep only recent history
        if len(state.performance_history) > 10:
            state.performance_history.pop(0)
        
        # Calculate average performance
        avg_performance = sum(state.performance_history) / len(state.performance_history)
        
        # Adapt strategy based on performance
        if avg_performance > state.high_performance_threshold:
            return HighPerformanceMode()
        elif avg_performance < state.low_performance_threshold:
            return SafeMode()
        else:
            return self  # Continue in current mode
```

## Transition Patterns

### 1. Linear Workflow

Simple sequential progression:

```python
# Step 1
class Initialize(Procedure[WorkflowState]):
    @handler
    def handle_initialization_complete(self, state: WorkflowState, event: InitComplete):
        return CollectData()

# Step 2
class CollectData(Procedure[WorkflowState]):
    @handler
    def handle_data_collected(self, state: WorkflowState, event: DataCollected):
        return ProcessData()

# Step 3
class ProcessData(Procedure[WorkflowState]):
    @handler
    def handle_processing_complete(self, state: WorkflowState, event: ProcessingComplete):
        return GenerateReport()

# Step 4
class GenerateReport(Procedure[WorkflowState]):
    @handler
    def handle_report_generated(self, state: WorkflowState, event: ReportGenerated):
        return End()
```

### 2. Branching Workflow

Different paths based on conditions:

```python
class DecisionPoint(Procedure[DecisionState]):
    @handler
    def handle_evaluation_result(self, state: DecisionState, event: EvaluationResult):
        result = event.payload
        
        if result.confidence > 0.9:
            return HighConfidencePath()
        elif result.confidence > 0.5:
            return MediumConfidencePath()
        else:
            return LowConfidencePath()

class HighConfidencePath(Procedure[DecisionState]):
    @handler
    def handle_high_confidence_processing(self, state: DecisionState, event: ProcessingResult):
        # Fast processing for high confidence
        return QuickFinish()

class MediumConfidencePath(Procedure[DecisionState]):
    @handler
    def handle_medium_confidence_processing(self, state: DecisionState, event: ProcessingResult):
        # Additional validation for medium confidence
        return ValidationStep()

class LowConfidencePath(Procedure[DecisionState]):
    @handler
    def handle_low_confidence_processing(self, state: DecisionState, event: ProcessingResult):
        # Human review for low confidence
        return HumanReviewStep()
```

### 3. Loop Patterns

Iterative processing with exit conditions:

```python
class IterativeProcessor(Procedure[IterativeState]):
    def __init__(self):
        self.iteration_count = 0
        self.max_iterations = 10
    
    @handler
    def handle_iteration_result(self, state: IterativeState, event: IterationResult):
        self.iteration_count += 1
        result = event.payload
        
        # Check convergence
        if result.converged:
            return ConvergedProcedure()
        
        # Check iteration limit
        if self.iteration_count >= self.max_iterations:
            return MaxIterationsReached()
        
        # Check improvement
        if result.improvement < state.min_improvement_threshold:
            return InsufficientImprovement()
        
        # Continue iterating
        state.current_value = result.new_value
        return self
```

### 4. Parallel Processing Coordination

Coordinate multiple parallel operations:

```python
class ParallelCoordinator(Procedure[ParallelState]):
    def __init__(self):
        self.completed_workers = set()
        self.expected_workers = {"worker_1", "worker_2", "worker_3"}
    
    @handler
    def handle_worker_complete(self, state: ParallelState, event: WorkerComplete):
        worker_id = event.payload.worker_id
        
        if event.payload.success:
            state.worker_results[worker_id] = event.payload.result
            self.completed_workers.add(worker_id)
        else:
            return f"Worker {worker_id} failed: {event.payload.error}"
        
        # Check if all workers completed
        if self.completed_workers >= self.expected_workers:
            return AggregateResults()
        
        return self
    
    @handler
    def handle_worker_failed(self, state: ParallelState, event: WorkerFailed):
        worker_id = event.payload.worker_id
        
        # Restart failed worker or abort
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            return RestartWorker(worker_id=worker_id)
        else:
            return AbortParallelProcessing()
```

## Error Handling in Handlers

### 1. Assertion-Based Validation

Use assertions for business logic validation:

```python
@handler
def handle_payment_processing(self, state: PaymentState, event: PaymentResult):
    payment = event.payload
    
    # Business logic assertions
    assert payment.amount > 0, "Payment amount must be positive"
    assert payment.currency in ["USD", "EUR", "GBP"], f"Unsupported currency: {payment.currency}"
    assert state.account_balance >= payment.amount, "Insufficient funds"
    
    # Process successful payment
    state.account_balance -= payment.amount
    state.transaction_history.append(payment)
    
    return PaymentConfirmation()
```

### 2. Exception Handling

Handle exceptions gracefully:

```python
@handler
def handle_external_api_call(self, state: APIState, event: APICallResult):
    try:
        result = event.payload
        
        # Validate API response
        if not result.success:
            if result.status_code == 429:  # Rate limited
                return "API rate limit exceeded. Please wait and retry."
            elif result.status_code >= 500:  # Server error
                return "API server error. Please retry the operation."
            else:
                return f"API call failed: {result.error_message}"
        
        # Process successful response
        state.api_data = result.data
        return ProcessAPIData()
        
    except Exception as e:
        return f"Unexpected error processing API response: {str(e)}"
```

### 3. Graceful Degradation

Provide fallback behavior when possible:

```python
@handler
def handle_service_unavailable(self, state: ServiceState, event: ServiceUnavailable):
    service_name = event.payload.service_name
    
    # Try fallback services
    if service_name == "primary_service" and state.fallback_available:
        state.using_fallback = True
        return TryFallbackService()
    
    # Use cached data if available
    elif state.has_cached_data and not state.cache_expired:
        state.using_cached_data = True
        return ProcessCachedData()
    
    # Graceful degradation with reduced functionality
    elif state.can_operate_with_reduced_functionality:
        state.reduced_mode = True
        return ReducedFunctionalityMode()
    
    # Complete failure
    else:
        return "All services unavailable and no fallback options. Please try again later."
```

## File System Example Handler Patterns

The file system operator demonstrates several sophisticated handler patterns:

### 1. Validation and State Updates

```python
class BackUpCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        
        src, dst = event.payload
        
        # Validation
        assert dst.startswith(sigma.backup_path), f"Destination {dst} is not from the backup path {sigma.backup_path}."
        
        # State updates
        self.copied.add(src)
        self.kv[dst] = src
        sigma.path_backuped.add(src)
        sigma.files_in_backup.add(dst)
        
        # Conditional transition
        if cond(sigma.path_read == sigma.path_backuped, "all files copied to backup location"):
            return MigrateCopy(prompt=make_prompt("For each file in the directory, copy it to the migrate location."))
        
        return self
```

### 2. Error Recovery Transitions

```python
class Copy(Procedure[FSInfo]):
    @handler
    def handle_eCopyRequest(self, sigma: FSInfo, event: eCopyRequest):
        if cond(self.retry == 0, "retry limit is reached"):
            return Abort()  # Transition to cleanup procedure
        
        src, dst = event.payload
        
        # Validation assertions that trigger recovery if failed
        assert src.startswith(sigma.src_path), f"Source {src} is not from the source path {sigma.src_path}."
        assert src in sigma.path_read, f"Source {src} not in read paths."
        
        return self
```

### 3. Comprehensive Cleanup Logic

```python
class Abort(Procedure[FSInfo]):
    @handler
    def handle_eFileDeleted(self, sigma: FSInfo, event: eFileDeleted):
        dst = event.payload
        if dst is None:
            return f"Failed to delete file {dst}. Please try again."
        
        # Track deletion by category
        if dst in sigma.files_in_backup:
            self.deleted_backup.add(dst)
        elif dst in sigma.files_in_migration:
            self.deleted_migration.add(dst)
        else:
            return f"File {dst} is not from the backup or migration path. Cannot delete."
        
        # Complex completion condition
        if cond(self.deleted_backup == sigma.files_in_backup and 
                self.deleted_migration == sigma.files_in_migration,
                "backups and partially migrated files are deleted"):
            return End()
        
        return self
```

## Best Practices

### 1. Keep Handlers Focused

Each handler should have a single responsibility:

```python
# Good - Focused handler
@handler
def handle_user_authenticated(self, state: AuthState, event: UserAuthenticated):
    state.current_user = event.payload.user
    state.session_token = event.payload.token
    return LoadUserPreferences()

# Avoid - Handler doing too much
@handler
def handle_user_authenticated(self, state: AuthState, event: UserAuthenticated):
    # Authentication logic
    state.current_user = event.payload.user
    state.session_token = event.payload.token
    
    # Preference loading logic
    preferences = load_preferences(event.payload.user.id)
    state.user_preferences = preferences
    
    # Analytics logic
    track_user_login(event.payload.user.id)
    
    # Email logic
    send_welcome_email(event.payload.user.email)
    
    return Dashboard()  # Too many responsibilities
```

### 2. Use Descriptive Condition Messages

Provide clear descriptions for conditional logic:

```python
# Good - Clear condition descriptions
@handler
def handle_processing_result(self, state: ProcessingState, event: ProcessingResult):
    if cond(event.payload.success and state.quality_score > 0.8, 
            "processing succeeded with high quality"):
        return AcceptResult()
    elif cond(event.payload.success and state.quality_score > 0.5,
              "processing succeeded with acceptable quality"):
        return ReviewResult()
    else:
        return "Processing failed or quality too low. Please retry with different parameters."

# Avoid - Unclear conditions
@handler
def handle_processing_result(self, state: ProcessingState, event: ProcessingResult):
    if cond(event.payload.success and state.quality_score > 0.8, "condition met"):
        return AcceptResult()
    # ...
```

### 3. Handle All Expected Events

Ensure handlers cover all possible event scenarios:

```python
class ComprehensiveProcedure(Procedure[MyState]):
    @handler
    def handle_success_event(self, state: MyState, event: SuccessEvent):
        return NextProcedure()
    
    @handler
    def handle_failure_event(self, state: MyState, event: FailureEvent):
        return ErrorProcedure()
    
    @handler
    def handle_timeout_event(self, state: MyState, event: TimeoutEvent):
        return TimeoutProcedure()
    
    @handler
    def handle_cancellation_event(self, state: MyState, event: CancellationEvent):
        return CancellationProcedure()
```

### 4. Validate State Consistency

Add validation to ensure state remains consistent:

```python
@handler
def handle_state_update(self, state: MyState, event: StateUpdate):
    # Update state
    state.value = event.payload.new_value
    
    # Validate consistency
    if state.value < 0:
        return "Invalid state: value cannot be negative"
    
    if state.value > state.max_allowed_value:
        return f"Invalid state: value {state.value} exceeds maximum {state.max_allowed_value}"
    
    return NextProcedure()
```

## Summary

Effective handler design in SOPy requires:

- **Clear return types**: Use procedures for transitions, strings for errors, self for staying
- **Focused responsibility**: Each handler should handle one type of event processing
- **Comprehensive coverage**: Handle all expected events and edge cases
- **State validation**: Ensure state consistency through assertions and validation
- **Descriptive conditions**: Use clear, human-readable condition descriptions
- **Error handling**: Gracefully handle exceptions and provide meaningful error messages

Handlers are the heart of SOPy workflows - they determine how your agent responds to events and progresses through the workflow. Well-designed handlers make your SOPs reliable, maintainable, and easy to understand.

## Next Steps

- [Automatic Prompts](05-automatic-prompts.md) - See how handlers influence prompt generation
- [Error Recovery](06-error-recovery.md) - Advanced error handling strategies
- [File System Example](08-file-system-example.md) - Complex handler patterns in action
