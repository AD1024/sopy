# Procedures and Events: The Building Blocks of SOPy

This tutorial dives deep into the two fundamental components of SOPy: Procedures and Events. Understanding these concepts is crucial for building effective SOPs.

## Procedures: States with Purpose

A `Procedure` represents a state in your SOP state machine. Each procedure defines:
- **What the agent should do** (via the prompt)
- **How to respond to events** (via handlers)
- **Local state and logic** (via instance variables)

### Basic Procedure Structure

```python
from sopy import Procedure, handler, make_prompt
from typing import Generic, TypeVar

# Define your state type
StateType = TypeVar('StateType')

class MyProcedure(Procedure[StateType]):
    # Required: Natural language description of what to do
    prompt = make_prompt("Describe what the agent should accomplish in this state")
    
    # Optional: Events to ignore (won't trigger handlers)
    ignore = [SomeEvent, AnotherEvent]
    
    # Optional: Instance variables for procedure-specific state
    def __init__(self):
        self.local_counter = 0
        self.attempts = 0
    
    @handler
    def handle_some_event(self, state: StateType, event: SomeEvent):
        # Handler logic here
        return NextProcedure()
```

### Procedure Lifecycle

1. **Creation**: Procedure instance is created
2. **Activation**: SOP transitions to this procedure
3. **Event Processing**: Handlers respond to incoming events
4. **Transition**: Handler returns next procedure or error message
5. **Deactivation**: SOP moves to next procedure

### Example: File Processing Procedure

```python
from sopy import Procedure, handler, make_prompt, End
from pydantic import BaseModel
from typing import List

class FileState(BaseModel):
    files_to_process: List[str]
    processed_files: List[str]
    failed_files: List[str]

class ProcessFiles(Procedure[FileState]):
    prompt = make_prompt("Process each file in the queue one by one")
    
    def __init__(self):
        self.current_file = None
        self.retry_count = 0
        self.max_retries = 3
    
    @handler
    def handle_file_processed(self, state: FileState, event: FileProcessed):
        if event.payload.success:
            # File processed successfully
            state.processed_files.append(self.current_file)
            self.retry_count = 0
            
            # Check if more files to process
            if state.files_to_process:
                self.current_file = state.files_to_process.pop(0)
                return self  # Stay in same state, process next file
            else:
                return End()  # All files processed
        else:
            # File processing failed
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                state.failed_files.append(self.current_file)
                return f"Failed to process {self.current_file} after {self.max_retries} attempts"
            else:
                return f"Processing failed for {self.current_file}, retrying ({self.retry_count}/{self.max_retries})"
```

## Events: Messages That Drive Transitions

Events represent things that happen in your system. They carry data (payload) and metadata about what occurred.

### Event Structure

```python
from sopy import Event
from typing import Generic, TypeVar, Optional, Any

PayloadType = TypeVar('PayloadType')

class MyEvent(Event[PayloadType]):
    """Custom event with typed payload"""
    
    def __init__(self, 
                 payload: PayloadType,
                 description: Optional[str] = None,
                 sender: Any = None,
                 target: Any = None,
                 toolUseId: Optional[str] = None):
        super().__init__(payload, description, sender, target, toolUseId)
```

### Event Categories

#### 1. Tool Events
Events triggered by tool usage (most common in agent workflows):

```python
class FileReadEvent(Event[tuple[str, str | None]]):
    """Event triggered when a file is read"""
    pass

class APICallEvent(Event[dict]):
    """Event triggered when an API call completes"""
    pass

class DatabaseQueryEvent(Event[list]):
    """Event triggered when a database query returns results"""
    pass
```

#### 2. User Interaction Events
Events from user input or interaction:

```python
class UserInputEvent(Event[str]):
    """User provided text input"""
    pass

class UserSelectionEvent(Event[int]):
    """User selected an option from a menu"""
    pass

class UserApprovalEvent(Event[bool]):
    """User approved or rejected an action"""
    pass
```

#### 3. System Events
Internal system events:

```python
class TimerExpiredEvent(Event[float]):
    """A timer has expired"""
    pass

class ResourceAvailableEvent(Event[str]):
    """A resource became available"""
    pass

class ErrorOccurredEvent(Event[Exception]):
    """An error occurred in the system"""
    pass
```

### Event Best Practices

#### 1. Use Descriptive Names
```python
# Good
class FileUploadCompleted(Event[str]): pass
class UserAuthenticationFailed(Event[str]): pass

# Avoid
class Event1(Event[str]): pass
class E(Event[str]): pass
```

#### 2. Include Rich Payloads
```python
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    success: bool
    message: str
    data: dict
    duration: float

class TaskCompleted(Event[ProcessingResult]): pass
```

#### 3. Add Metadata When Useful
```python
# Event with sender information
event = FileProcessed(
    payload=result,
    sender="file_processor_agent",
    description="File processing completed successfully"
)
```

## Handlers: Connecting Events to Actions

Handlers are the bridge between events and state transitions. They define how procedures respond to events.

### Handler Patterns

#### 1. Simple Transition
```python
@handler
def handle_task_complete(self, state: MyState, event: TaskComplete):
    if event.payload.success:
        return NextProcedure()
    else:
        return "Task failed, please retry"
```

#### 2. Conditional Logic
```python
@handler
def handle_user_input(self, state: MyState, event: UserInput):
    user_text = event.payload.lower()
    
    if "yes" in user_text or "approve" in user_text:
        state.approved = True
        return ExecuteAction()
    elif "no" in user_text or "reject" in user_text:
        state.approved = False
        return CancelAction()
    else:
        return "Please respond with 'yes' or 'no'"
```

#### 3. State Accumulation
```python
@handler
def handle_data_received(self, state: MyState, event: DataReceived):
    # Accumulate data
    state.collected_data.append(event.payload)
    
    # Check if we have enough data
    if len(state.collected_data) >= state.required_count:
        return ProcessData()
    else:
        return self  # Stay in current state, wait for more data
```

#### 4. Error Handling
```python
@handler
def handle_operation_result(self, state: MyState, event: OperationResult):
    if event.payload.error:
        self.error_count += 1
        if self.error_count > self.max_errors:
            return AbortProcedure()
        else:
            return f"Operation failed: {event.payload.error}. Retrying..."
    
    # Success case
    state.result = event.payload.data
    return NextProcedure()
```

## Advanced Patterns

### 1. Procedure Inheritance
```python
class BaseCopyProcedure(Procedure[FileState]):
    """Base class for file copying procedures"""
    
    def __init__(self):
        self.retry_count = 0
        self.max_retries = 3
    
    def handle_copy_failure(self, state: FileState, event: CopyFailed):
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            return "Copy operation failed after maximum retries"
        return f"Copy failed, retrying ({self.retry_count}/{self.max_retries})"

class BackupCopy(BaseCopyProcedure):
    prompt = make_prompt("Copy files to backup location")
    
    @handler
    def handle_copy_success(self, state: FileState, event: CopySuccess):
        state.backup_files.add(event.payload.destination)
        return MigrateCopy()

class MigrateCopy(BaseCopyProcedure):
    prompt = make_prompt("Copy files to final destination")
    
    @handler
    def handle_copy_success(self, state: FileState, event: CopySuccess):
        state.migrated_files.add(event.payload.destination)
        return CleanupProcedure()
```

### 2. Dynamic Procedure Creation
```python
@handler
def handle_task_list(self, state: MyState, event: TaskListReceived):
    tasks = event.payload
    
    if not tasks:
        return End()
    
    # Create procedure with dynamic prompt based on first task
    first_task = tasks[0]
    return ProcessTask(
        prompt=make_prompt(f"Process task: {first_task.description}"),
        task=first_task
    )
```

### 3. Event Filtering
```python
class SelectiveProcessor(Procedure[MyState]):
    # Ignore certain events
    ignore = [LowPriorityEvent, DebugEvent]
    
    @handler
    def handle_high_priority(self, state: MyState, event: HighPriorityEvent):
        # This will be processed
        return ProcessUrgentTask()
    
    @handler
    def handle_low_priority(self, state: MyState, event: LowPriorityEvent):
        # This will never be called due to ignore list
        pass
```

## Real-World Example: File System Operator

Let's examine how the file system operator uses procedures and events:

### Events
```python
class eReadDirectory(Event[Tuple[str, List[str] | None]]): 
    """Directory read completed"""
    pass

class eCopyRequest(Event[Tuple[str, str]]):
    """Request to copy a file"""
    def __init__(self, toolUseId: str, source: str, dest: str):
        super().__init__(toolUseId=toolUseId, payload=(source, dest))

class eFileCopied(Event[Tuple[str, str] | None]): 
    """File copy operation completed"""
    pass
```

### Procedures
```python
class ReadDir(Procedure[FSInfo]):
    prompt = make_prompt("Read the directory to be migrated")

    @handler
    def handle_eReadDirectory(self, sigma: FSInfo, event: eReadDirectory):
        if event.payload[1] is None:
            return f"Failed to read directory {event.payload[0]}. Please check the path and try again."
        
        # Update state with discovered files
        sigma.path_read |= set((f'{event.payload[0]}/{filename}' for filename in event.payload[1]))
        sigma.src_path = event.payload[0]
        
        # Transition to backup phase
        return BackUpCopy(prompt=make_prompt("For each file in the directory, copy it to the given backup location."))
```

## Testing Procedures and Events

### Unit Testing Procedures
```python
import unittest
from unittest.mock import Mock

class TestMyProcedure(unittest.TestCase):
    def setUp(self):
        self.state = MyState()
        self.procedure = MyProcedure()
    
    def test_successful_event_handling(self):
        event = SuccessEvent(payload="test_data")
        result = self.procedure.handle_success_event(self.state, event)
        
        self.assertIsInstance(result, NextProcedure)
        self.assertEqual(self.state.processed_data, "test_data")
    
    def test_error_recovery(self):
        event = FailureEvent(payload="error_message")
        result = self.procedure.handle_failure_event(self.state, event)
        
        self.assertIsInstance(result, str)
        self.assertIn("error_message", result)
```

### Integration Testing with Events
```python
def test_procedure_flow():
    from sopy.core.event import EventBus
    
    sop = SOP(state=MyState(), init_proc=StartProcedure())
    
    with sop:
        # Trigger events and verify state transitions
        EventBus.push_event(StartEvent(payload="begin"))
        assert isinstance(sop.current_proc, ProcessingProcedure)
        
        EventBus.push_event(CompleteEvent(payload="done"))
        assert isinstance(sop.current_proc, End)
```

## Summary

Procedures and Events form the foundation of SOPy:

- **Procedures** define states and behavior through prompts and handlers
- **Events** carry information about what happened in the system
- **Handlers** connect events to state transitions and actions
- **Type safety** ensures reliable event processing
- **Patterns** like inheritance and dynamic creation enable complex workflows

Understanding these concepts deeply will help you build robust, maintainable SOPs that can handle complex agent workflows reliably.

## Next Steps

- [State Management](03-state-management.md) - Learn about the Sigma pattern and state handling
- [Handlers and Transitions](04-handlers-and-transitions.md) - Advanced handler patterns
- [File System Example](08-file-system-example.md) - See these concepts in action
