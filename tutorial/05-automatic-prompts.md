# Automatic Prompt Generation: From State Machines to Natural Language

One of SOPy's most powerful features is its ability to automatically generate natural language Standard Operating Procedure (SOP) prompts from your state machine definitions. This tutorial explains how this works and how to leverage it effectively.

## Why Automatic Prompt Generation?

Traditional approaches require manually writing and maintaining both:
1. **Code logic** - The actual workflow implementation
2. **Natural language prompts** - Instructions for the LLM agent

This creates several problems:
- **Inconsistency** - Code and prompts can drift apart
- **Maintenance burden** - Changes require updating both code and prompts
- **Error-prone** - Manual synchronization leads to mistakes
- **Verification difficulty** - Hard to ensure prompts match actual behavior

SOPy solves this by generating prompts directly from your state machine, ensuring consistency.

## How It Works

SOPy's prompt generator analyzes your procedure definitions using:

1. **AST (Abstract Syntax Tree) Analysis** - Parses your handler methods
2. **Control Flow Analysis** - Understands conditional logic and transitions
3. **Type Introspection** - Extracts meaningful information from type hints
4. **Natural Language Generation** - Converts logic to readable instructions

### The Generation Process

```python
from sopy.prompt.prompt_generator import make_prompt_from_procedure

# Your state machine definition
entry_procedure = ReadDir()

# Automatic prompt generation
objective = "Migrate files from source to destination with backup safety"
generated_prompt = make_prompt_from_procedure(entry_procedure, objective)

print(generated_prompt)
```

## Example: File System Operator

Let's see how the file system operator's state machine becomes a natural language SOP.

### State Machine Definition

```python
class ReadDir(Procedure[FSInfo]):
    prompt = make_prompt("Read the directory to be migrated")

    @handler
    def handle_eReadDirectory(self, sigma: FSInfo, event: eReadDirectory):
        if event.payload[1] is None:
            return f"Failed to read directory {event.payload[0]}. Please check the path and try again."
        sigma.path_read |= set((f'{event.payload[0]}/{filename}' for filename in event.payload[1]))
        sigma.src_path = event.payload[0]
        return BackUpCopy(prompt=make_prompt("For each file in the directory, copy it to the given backup location."))

class BackUpCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        src, dst = event.payload
        assert dst.startswith(sigma.backup_path), f"Destination {dst} is not from the backup path {sigma.backup_path}."
        self.copied.add(src)
        sigma.path_backuped.add(src)
        sigma.files_in_backup.add(dst)
        if cond(sigma.path_read == sigma.path_backuped, "all files copied to backup location"):
            return MigrateCopy(prompt=make_prompt("For each file in the directory, copy it to the migrate location."))
        return self
... # full definitions see `file_sys_sop.py`
```

### Generated Natural Language SOP

```
Here is the standard operating procedure (SOP) for Migrate files from source to destination with backup safety:

Procedure ReadDir:
    Instruction: Read the directory to be migrated. Then proceed to BackUpCopy.
    Action: eReadDirectory

Procedure BackUpCopy:
    Instruction: For each file in the directory, copy it to the given backup location. If retry limit is reached, then go to Abort. If all files copied to backup location, then go to MigrateCopy.
    Action: eFileCopied, eCopyRequest

Procedure MigrateCopy:
    Instruction: For each file in the directory, copy it to the migrate location. If retry limit is reached, then go to Abort. If all files copied to the migration location, then go to Clear.
    Action: eFileCopied, eCopyRequest

Procedure Abort:
    Instruction: Delete all copied backup files and migrated files (if any), and abort the procedure. If backups and partially migrated files are deleted, then finish the procedure, and report related results to the user.
    Action: eFileDeleted, eFileDeleteRequest

Procedure Clear:
    Instruction: Delete the backup files and the original files (not the original directory) and finish the procedure. If all files deleted from backup location, then finish the procedure, and report related results to the user.
    Action: eFileDeleted, eFileDeleteRequest
```

## Understanding the Generation Components

### 1. Procedure Analysis

The generator extracts information from each procedure:

```python
class ProcedureNode:
    procedure: Type[Procedure]      # The procedure class
    name: str                       # Procedure name
    prompt_text: str               # The prompt content
    handlers: Dict[Type[Event], HandlerInfo]  # Event handlers
    is_terminal: bool              # Whether this ends the SOP
    docstring: Optional[str]       # Documentation
```

### 2. Handler Analysis

For each handler, the generator analyzes:

```python
class HandlerInfo:
    method_name: str               # Handler method name
    event_type: Type[Event]        # Event it handles
    ctx: Dict[int, ast.expr]       # Variable context
    returns_procedure: bool        # Can return next procedure
    returns_string: bool           # Can return error message
    docstring: Optional[str]       # Handler documentation
```

### 3. Control Flow Analysis

The generator understands conditional logic:

```python
@handler
def handle_event(self, state: MyState, event: MyEvent):
    if cond(state.ready, "system is ready"):
        return NextProcedure()
    elif cond(state.error_count > 3, "too many errors"):
        return AbortProcedure()
    else:
        return "Please wait for system to be ready"
```

Becomes:
```
If system is ready, then go to NextProcedure.
If too many errors, then go to AbortProcedure.
Otherwise, return error message.
```

## Advanced Features

### 1. Conditional Description with `cond()`

The `cond()` function provides human-readable descriptions for conditions:

```python
from sopy.prompt import cond

@handler
def handle_file_processed(self, state: FileState, event: FileProcessed):
    if cond(len(state.processed_files) >= state.target_count, 
            "all required files have been processed"):
        return CompleteProcedure()
    return self
```

Generated text:
```
If all required files have been processed, then go to CompleteProcedure.
```

### 2. Pydantic Field Descriptions

The generator can extract field descriptions from Pydantic models:

```python
class FSInfo(BaseModel):
    src_path: str = Field(..., description="The source path from which files are to be migrated.")
    dst_path: str = Field(..., description="The destination path to which files are to be migrated.")
    files_in_backup: set[str] = Field(default_factory=set, description="Set of files that have been backed up to the backup directory.")
```

When these fields are referenced in conditions, the generator uses the descriptions:
```python
if cond(len(sigma.files_in_backup) == 0, "backup directory is empty"):
    return End()
```

Becomes:
```
If backup directory is empty, then finish the procedure.
```

### 3. Complex State Transitions

The generator handles various transition patterns:

#### Sequential Flow
```python
@handler
def handle_step_complete(self, state: MyState, event: StepComplete):
    return NextStep()
```
→ "Then proceed to NextStep."

#### Conditional Branching
```python
@handler
def handle_result(self, state: MyState, event: Result):
    if event.payload.success:
        return SuccessStep()
    else:
        return FailureStep()
```
→ "If successful, then go to SuccessStep. Otherwise, go to FailureStep."

#### Retry Patterns
```python
@handler
def handle_attempt(self, state: MyState, event: Attempt):
    if event.payload.failed and self.retry_count < self.max_retries:
        self.retry_count += 1
        return self
    elif event.payload.failed:
        return AbortStep()
    else:
        return SuccessStep()
```
→ "If attempt failed and retry limit not reached, try again. If retry limit reached, then go to AbortStep. If successful, then go to SuccessStep."

## Customizing Generated Prompts

### 1. Procedure Prompts

Each procedure's prompt becomes the main instruction:

```python
class ProcessData(Procedure[MyState]):
    prompt = make_prompt("Analyze the collected data and generate insights")
```

### 2. Handler Documentation

Add docstrings to handlers for additional context:

```python
@handler
def handle_data_ready(self, state: MyState, event: DataReady):
    """Process data when it becomes available and meets quality criteria."""
    if self.validate_data(event.payload):
        return GenerateReport()
    return "Data quality insufficient, please collect more data"
```

### 3. Event Descriptions

Provide meaningful event descriptions:

```python
class DataValidated(Event[bool]):
    """Event triggered when data validation completes"""
    
    def __init__(self, payload: bool, description: str = "Data validation completed"):
        super().__init__(payload, description=description)
```

## Best Practices for Prompt Generation

### 1. Write Clear Procedure Prompts

```python
# Good - Clear and specific
class BackupFiles(Procedure[FileState]):
    prompt = make_prompt("Create backup copies of all files in the source directory")

# Avoid - Vague or unclear
class DoStuff(Procedure[FileState]):
    prompt = make_prompt("Do the thing")
```

### 2. Use Descriptive Condition Messages

```python
# Good - Explains the business logic
if cond(all_files_processed and no_errors_occurred, 
        "all files successfully processed without errors"):
    return CompleteTask()

# Avoid - Technical but not explanatory
if cond(len(errors) == 0 and processed == total, "condition met"):
    return CompleteTask()
```

### 3. Structure Complex Logic Clearly

```python
# Good - Clear conditional structure
@handler
def handle_processing_result(self, state: ProcessState, event: ProcessingResult):
    if cond(event.payload.success, "processing completed successfully"):
        state.completed_items += 1
        if cond(state.completed_items >= state.total_items, "all items processed"):
            return FinalizeProcedure()
        else:
            return self  # Continue processing
    else:
        if cond(self.retry_count < self.max_retries, "retry limit not exceeded"):
            self.retry_count += 1
            return "Processing failed, retrying..."
        else:
            return AbortProcedure()
```

### 4. Provide Rich State Descriptions

Use Pydantic models with detailed field descriptions:

```python
class TaskState(BaseModel):
    current_task: str = Field(..., description="The task currently being processed")
    completed_tasks: List[str] = Field(default_factory=list, description="List of successfully completed tasks")
    failed_tasks: List[str] = Field(default_factory=list, description="List of tasks that failed processing")
    retry_count: int = Field(default=0, description="Number of retry attempts for the current task")
```

## Testing Generated Prompts

### 1. Verify Prompt Generation

```python
def test_prompt_generation():
    entry_procedure = MyStartProcedure()
    objective = "Complete the workflow successfully"
    
    prompt = make_prompt_from_procedure(entry_procedure, objective)
    
    # Verify key elements are present
    assert "Procedure MyStartProcedure:" in prompt
    assert objective in prompt
    assert "Action:" in prompt
```

### 2. Cross-Check with Implementation

```python
def test_prompt_matches_implementation():
    # Generate prompt
    prompt = make_prompt_from_procedure(StartProcedure(), "Test objective")
    
    # Verify that all procedures mentioned in prompt exist
    procedure_names = extract_procedure_names(prompt)
    for name in procedure_names:
        assert hasattr(sys.modules[__name__], name), f"Procedure {name} not found"
```

## Integration with Agents

The generated prompts work seamlessly with Strands agents:

```python
from sopy.interface.strands import StrandsAgent

# Generate the SOP prompt
sop_prompt = make_prompt_from_procedure(entry_procedure, objective)

# Create agent with generated prompt
agent = StrandsAgent(
    tools=[read_dir, cp, delete],
    system_prompt=sop_prompt
)

# The agent now follows the exact SOP defined by your state machine
with sop:
    agent("Execute the file migration procedure")
```

## Summary

Automatic prompt generation in SOPy:

- **Ensures consistency** between code and prompts
- **Reduces maintenance** by generating prompts from code
- **Improves reliability** through automated synchronization
- **Enables verification** by making behavior explicit
- **Supports complex logic** with conditional descriptions
- **Integrates seamlessly** with LLM agents

The key is to write clear, well-structured procedures with meaningful prompts and condition descriptions. SOPy handles the complex task of converting your state machine logic into natural language instructions that LLM agents can follow reliably.

## Next Steps

- [Error Recovery](06-error-recovery.md) - Learn about recovery mechanisms
- [Tool Integration](07-tool-integration.md) - Connect with Strands agents
- [File System Example](08-file-system-example.md) - See prompt generation in action
