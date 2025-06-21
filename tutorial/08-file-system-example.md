# File System Example: A Complete SOPy Walkthrough

The file system operator example in `examples/file_system_operator/` demonstrates a real-world use case of SOPy: safely migrating files from one directory to another with backup protection and automatic recovery. This tutorial walks through the complete implementation to show how all SOPy concepts work together.

## Overview: The File Migration Problem

The task is to migrate files from a source directory to a destination directory while:
1. **Creating backups** before any destructive operations
2. **Handling failures gracefully** with automatic rollback
3. **Ensuring atomicity** - either all files migrate or none do
4. **Providing observability** through events and logging

This is a perfect example for SOPy because it involves:
- **Multiple sequential steps** that must be coordinated
- **Error conditions** that require different recovery strategies
- **State tracking** across multiple operations
- **Tool integration** with file system operations

## Architecture Overview

The file system operator implements a 5-stage pipeline:

```
ReadDir → BackUpCopy → MigrateCopy → Clear → End
    ↓         ↓           ↓          ↓
  Abort ←── Abort ←──── Abort ←──── (success)
```

Each stage has specific responsibilities and failure modes.

## State Management: FSInfo

The entire operation is coordinated through a shared state object:

```python
from pydantic import Field, BaseModel

class FSInfo(BaseModel):
    src_path: str = Field(..., description="The source path from which files are to be migrated.")
    dst_path: str = Field(..., description="The destination path to which files are to be migrated.")
    backup_path: str = Field(..., description="The backup path where files are temporarily stored during migration.")
    
    # Tracking sets for different stages
    path_read: set[str] = Field(default_factory=set, description="Set of paths that have been read from the source directory.")
    path_copied: set[str] = Field(default_factory=set, description="Set of paths that have been copied to the destination directory.")
    path_backuped: set[str] = Field(default_factory=set, description="Set of paths that have been backed up to the backup directory.")
    
    # File tracking for cleanup
    files_in_backup: set[str] = Field(default_factory=set, description="Set of files that have been backed up to the backup directory.")
    files_in_migration: set[str] = Field(default_factory=set, description="Set of files that have been migrated to the destination directory.")
```

The Pydantic model provides:
- **Type safety** for all state fields
- **Field descriptions** that are used in automatic prompt generation
- **Default values** for collection fields
- **Validation** of state updates

## Events: Modeling File System Operations

The system defines events for each file system operation:

```python
from typing import Tuple, List
from sopy import Event

class eReadDirectory(Event[Tuple[str, List[str] | None]]):
    """Directory read completed - payload is (path, file_list_or_none)"""
    pass

class eCopyRequest(Event[Tuple[str, str]]):
    """Request to copy a file - payload is (source, destination)"""
    def __init__(self, toolUseId: str, source: str, dest: str):
        super().__init__(toolUseId=toolUseId, payload=(source, dest))

class eFileCopied(Event[Tuple[str, str] | None]):
    """File copy operation completed - payload is (source, dest) or None if failed"""
    pass

class eFileDeleteRequest(Event[str]):
    """Request to delete a file - payload is file path"""
    def __init__(self, toolUseId: str, path: str):
        super().__init__(toolUseId=toolUseId, payload=path)

class eFileDeleted(Event[str | None]):
    """File deletion completed - payload is path or None if failed"""
    pass
```

Key design decisions:
- **Typed payloads** ensure type safety
- **Request/Response pairs** for tool operations
- **Failure representation** using `None` payloads
- **Tool integration** through `toolUseId` parameter

## Procedures: The State Machine

### 1. ReadDir: Discovery Phase

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
        
        Log.info(f"Directory read {sigma.src_path} with files: {' | '.join(sigma.path_read)}")
        
        # Transition to backup phase
        return BackUpCopy(prompt=make_prompt("For each file in the directory, copy it to the given backup location."))
```

**Key patterns:**
- **Error handling**: Check for `None` payload indicating failure
- **State updates**: Add discovered files to tracking set
- **Logging**: Provide observability into the process
- **Dynamic transitions**: Create next procedure with specific prompt

### 2. Copy: Base Class for File Operations

```python
class Copy(Procedure[FSInfo]):
    def __init__(self, prompt: Prompt):
        self.prompt = prompt
        self.copied = set()      # Files copied in this operation
        self.srcs = set()        # Source files being processed
        self.kv = dict()         # Source-to-destination mapping
        self.retry = 2           # Retry attempts remaining

    @handler
    def handle_eCopyRequest(self, sigma: FSInfo, event: eCopyRequest):
        if cond(self.retry == 0, "retry limit is reached"):
            return Abort()
        
        src, dst = event.payload
        
        # Validation assertions
        assert src.startswith(sigma.src_path), f"Source {src} is not from the source path {sigma.src_path}."
        assert src in sigma.path_read, f"Source {src} not in read paths."
        assert dst not in self.copied, f"File {dst} already exists in this backup operation for {self.kv[dst]}."
        assert src not in self.srcs, f"File {src} already copied in this backup operation."
        
        return self
```

**Design patterns:**
- **Inheritance**: Base class for common copy logic
- **Local state**: Procedure-specific tracking variables
- **Retry logic**: Built-in retry mechanism with limits
- **Validation**: Assertions ensure operation correctness
- **Conditional descriptions**: `cond()` provides human-readable conditions

### 3. BackUpCopy: Safety First

```python
class BackUpCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        
        src, dst = event.payload
        
        # Validate backup destination
        assert dst.startswith(sigma.backup_path), f"Destination {dst} is not from the backup path {sigma.backup_path}."
        
        # Update tracking state
        self.copied.add(src)
        self.kv[dst] = src
        sigma.path_backuped.add(src)
        sigma.files_in_backup.add(dst)
        
        Log.info(f"Current path_backuped: {sigma.path_backuped}, copied files: {self.copied}")
        
        # Check completion condition
        if cond(sigma.path_read == sigma.path_backuped, "all files copied to backup location"):
            return MigrateCopy(prompt=make_prompt("For each file in the directory, copy it to the migrate location."))
        
        return self
```

**Key features:**
- **Failure handling**: Retry on `None` payload
- **Path validation**: Ensure files go to correct backup location
- **Progress tracking**: Update multiple state collections
- **Completion detection**: Compare sets to determine when done
- **State transition**: Move to migration phase when backup complete

### 4. MigrateCopy: The Main Operation

```python
class MigrateCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        
        src, dst = event.payload
        
        # Update migration tracking
        self.copied.add(src)
        self.kv[dst] = src
        sigma.files_in_migration.add(dst)
        
        Log.info(f"Current files_in_migration: {sigma.files_in_migration}, copied files: {self.copied}")
        
        # Check if migration complete
        if cond(sigma.path_read == self.copied, "all files copied to the migration location"):
            return Clear()
        
        return self
```

**Similar patterns to BackUpCopy but:**
- **Different validation**: No path restrictions for destination
- **Different completion condition**: Compare with `self.copied` instead of state set
- **Different next step**: Transition to cleanup phase

### 5. Clear: Success Cleanup

```python
class Clear(Procedure[FSInfo]):
    prompt = make_prompt("Delete the backup files and the original files (not the original directory) and finish the procedure.")

    def __init__(self):
        self.deleted = set()

    @handler
    def handle_eFileDeleteRequest(self, sigma: FSInfo, event: eFileDeleteRequest):
        dst = event.payload
        
        # Validation
        assert dst not in self.deleted, f"File {dst} already deleted in this backup operation."
        assert dst in sigma.files_in_backup or dst in sigma.path_read, f"File {dst} not in backup files. Cannot delete."
        
        return self

    @handler
    def handle_eFileDeleted(self, sigma: FSInfo, event: eFileDeleted):
        dst = event.payload
        if dst is None:
            return f"Failed to delete file {dst}. Please try again."
        
        # Update state
        sigma.files_in_backup.discard(dst)
        sigma.path_read.discard(dst)
        
        # Check completion
        if cond(len(sigma.files_in_backup) == 0 and len(sigma.path_read) == 0, "all files deleted from backup location"):
            return End()
        
        return self
```

**Cleanup patterns:**
- **Two-phase deletion**: Request validation, then execution
- **State cleanup**: Remove files from tracking sets
- **Completion detection**: Check that all tracked files are removed
- **Final transition**: Move to `End()` when complete

### 6. Abort: Failure Recovery

```python
class Abort(Procedure[FSInfo]):
    prompt = make_prompt("Delete all copied backup files and migrated files (if any), and abort the procedure")

    def __init__(self):
        self.deleted_backup = set()
        self.deleted_migration = set()

    @handler
    def handle_eFileDeleteRequest(self, sigma: FSInfo, event: eFileDeleteRequest):
        dst = event.payload
        
        # Determine which set this file belongs to
        if dst in sigma.files_in_backup:
            assert dst not in self.deleted_backup, f"File {dst} already deleted in the backup path."
        elif dst in sigma.files_in_migration:
            assert dst not in self.deleted_migration, f"File {dst} already deleted in the migration path."
        else:
            return f"File {dst} is not from the backup or migration path. Cannot delete."
        
        return self

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
        
        # Check if all cleanup complete
        if cond(self.deleted_backup == sigma.files_in_backup and self.deleted_migration == sigma.files_in_migration,
                "backups and partially migrated files are deleted"):
            return End()
        
        return self
```

**Recovery patterns:**
- **Comprehensive cleanup**: Handle both backup and migration files
- **Separate tracking**: Different sets for different file categories
- **Validation**: Ensure only appropriate files are deleted
- **Complete rollback**: Only finish when all created files are removed

## Tool Integration

The file system operations are implemented as Strands tools:

```python
from sopy.interface.strands import tool

@tool(eReadDirectory)
def read_dir(path: str):
    """Read the contents of `path` and return a list of file names."""
    node = _filesys._find_node(path)
    if node is None or node.is_file:
        return path, None
    return path, [child.name for child in node.children]

@tool(eFileCopied, eCopyRequest)
def cp(source: str, dest: str):
    """Copy a file from `source` to `dest`."""
    result = _filesys.copy(source, dest)
    if "Error" in result:
        return None
    return source, dest

@tool(eFileDeleted, eFileDeleteRequest)
def delete(path: str):
    """Delete a file or directory at `path`."""
    result = _filesys.delete(path)
    if "Error" in result:
        return None
    return path
```

**Tool patterns:**
- **Event decoration**: `@tool` decorator links tools to events
- **Request/Response**: Tools can trigger both request and response events
- **Error handling**: Return `None` to indicate failure
- **Simulation**: Uses in-memory file system for testing

## SOP Orchestration

The main SOP class ties everything together:

```python
class FileSystemOperator(SOP[FSInfo]):
    def __init__(self, src_path: str, dst_path: str, backup_path: str):
        super().__init__(
            state=FSInfo(src_path=src_path, dst_path=dst_path, backup_path=backup_path),
            init_proc=ReadDir()
        )

def start_task(src_path: str, dst_path: str, backup_path: str):
    # Create SOP with event observation
    sop = FileSystemOperator(src_path, dst_path, backup_path).observes(
        eReadDirectory, eFileDeleteRequest, eFileDeleted, eCopyRequest, eFileCopied
    )
    
    # Generate automatic prompt
    objective = "Migrate files from source to destination with backup safety"
    prompt = make_prompt_from_procedure(sop.init_proc, objective)
    print(prompt)
    
    # Create agent with tools and generated prompt
    agent = StrandsAgent(tools=[read_dir, cp, delete], system_prompt=prompt)
    
    # Add error recovery
    sop = sop.with_recovery(
        error_handler=lambda prompt: agent(f"Error occurred: {prompt}"),
        retry_limits=3
    )

    # Execute the SOP
    with sop:
        agent(f"Move all files from {src_path} to {dst_path} with backup at {backup_path} using given tools only.")
    
    # Validation
    migrated_files = read_dir(dst_path)[1]
    backup_files = read_dir(backup_path)[1]
    remaining_files = read_dir(src_path)[1]
    
    assert migrated_files == current_files, "Files in destination don't match source"
    assert len(backup_files) == 0, "Backup files should be empty after migration"
    assert remaining_files == [], "Source directory should be empty after migration"
```

**Integration patterns:**
- **Event observation**: Specify which events the SOP should handle
- **Automatic prompts**: Generate instructions from state machine
- **Agent integration**: Combine generated prompts with tools
- **Error recovery**: Provide custom error handling logic
- **Validation**: Verify the operation completed correctly

## Generated SOP Prompt

The automatic prompt generation produces:

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

This prompt perfectly captures the state machine logic in natural language that an LLM agent can follow.

## Key Design Principles Demonstrated

### 1. State-Driven Design
- **Centralized state**: All procedures operate on shared `FSInfo` state
- **State evolution**: Each procedure updates relevant state fields
- **State validation**: Assertions ensure state consistency

### 2. Event-Driven Architecture
- **Loose coupling**: Procedures respond to events, not direct calls
- **Type safety**: Events carry typed payloads
- **Tool integration**: Events bridge between tools and procedures

### 3. Error Recovery
- **Multiple strategies**: Retry vs. abort based on context
- **State cleanup**: Abort procedure removes all created files
- **Graceful degradation**: System can recover from partial failures

### 4. Observability
- **Event tracing**: All operations generate events
- **Logging**: Key state changes are logged
- **Validation**: Post-operation checks verify correctness

### 5. Composability
- **Procedure inheritance**: `Copy` base class shared by backup and migration
- **Dynamic creation**: Procedures created with context-specific prompts
- **Tool reuse**: Same tools used across different procedures

## Testing the File System Operator

The example includes comprehensive testing:

```python
def test_successful_migration():
    """Test complete successful migration"""
    start_task('/files', '/migrated_files', '/backups')
    
    # Verify final state
    assert read_dir('/migrated_files')[1] == ['file_0.txt', 'file_1.txt', 'file_2.txt']
    assert read_dir('/backups')[1] == []
    assert read_dir('/files')[1] == []

def test_failure_recovery():
    """Test that failures trigger proper cleanup"""
    # Simulate failure during migration
    # Verify that backup files are cleaned up
    pass
```

## Summary

The file system operator demonstrates how SOPy enables building complex, reliable agent workflows:

- **State machines** provide structure and predictability
- **Events** enable loose coupling and observability  
- **Automatic prompts** ensure consistency between code and instructions
- **Error recovery** handles failures gracefully
- **Tool integration** connects to external systems seamlessly

This pattern can be applied to many other domains: database operations, API workflows, data processing pipelines, and more. The key is to think in terms of states, events, and transitions while leveraging SOPy's automatic prompt generation and error recovery capabilities.

## Next Steps

- Try modifying the file system operator to add new features
- Apply similar patterns to your own domain problems
- Explore [Advanced Patterns](09-advanced-patterns.md) for more complex scenarios
- Review [Error Recovery](06-error-recovery.md) for more recovery strategies
