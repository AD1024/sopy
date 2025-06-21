# Tool Integration: Connecting SOPy with External Systems

SOPy's power comes from its ability to integrate seamlessly with external tools and systems. This tutorial explains how to connect SOPy with Strands agents, create custom tools, and build robust tool-based workflows.

## Overview: Tools in SOPy

Tools are the bridge between your SOPy state machine and the external world. They enable agents to:
- **Interact with APIs** - Make HTTP requests, call web services
- **Manipulate files** - Read, write, copy, delete files and directories
- **Query databases** - Execute SQL, retrieve and update data
- **Control systems** - Start/stop services, manage infrastructure
- **Process data** - Transform, analyze, and validate information

SOPy integrates with tools through an event-driven architecture where tool calls trigger events that drive state transitions.

## Strands Integration

SOPy provides first-class integration with the Strands agent framework through the `sopy.interface.strands` module.

### Basic Strands Agent Setup

```python
from sopy.interface.strands import StrandsAgent
from sopy import SOP, Procedure, handler, make_prompt

# Create your SOP
sop = SOP(state=MyState(), init_proc=StartProcedure())

# Generate automatic prompt
from sopy.prompt.prompt_generator import make_prompt_from_procedure
sop_prompt = make_prompt_from_procedure(sop.init_proc, "Complete the task")

# Create Strands agent with tools and generated prompt
agent = StrandsAgent(
    tools=[my_tool1, my_tool2, my_tool3],
    system_prompt=sop_prompt
)

# Run the workflow
with sop:
    result = agent("Execute the workflow using the available tools")
```

### The `@tool` Decorator

SOPy provides a `@tool` decorator that automatically connects Strands tools to SOPy events:

```python
from sopy.interface.strands import tool
from sopy import Event

# Define events
class FileReadEvent(Event[tuple[str, str | None]]):
    """Event triggered when a file is read"""
    pass

class FileWriteEvent(Event[bool]):
    """Event triggered when a file is written"""
    pass

# Create tools that trigger events
@tool(FileReadEvent)
def read_file(path: str):
    """Read the contents of a file"""
    try:
        with open(path, 'r') as f:
            content = f.read()
        return path, content
    except Exception as e:
        return path, None

@tool(FileWriteEvent)
def write_file(path: str, content: str):
    """Write content to a file"""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return True
    except Exception:
        return False
```

### Request/Response Tool Patterns

For operations that involve both a request and response, you can specify both events:

```python
class CopyRequest(Event[tuple[str, str]]):
    """Request to copy a file"""
    def __init__(self, toolUseId: str, source: str, dest: str):
        super().__init__(toolUseId=toolUseId, payload=(source, dest))

class CopyResult(Event[tuple[str, str] | None]):
    """Result of copy operation"""
    pass

@tool(CopyResult, CopyRequest)
def copy_file(source: str, dest: str):
    """Copy a file from source to destination"""
    try:
        import shutil
        shutil.copy2(source, dest)
        return source, dest
    except Exception:
        return None
```

## Tool Design Patterns

### 1. Simple Function Tools

Basic tools that perform a single operation:

```python
@tool(DatabaseQueryResult)
def query_database(sql: str):
    """Execute a SQL query and return results"""
    try:
        connection = get_database_connection()
        cursor = connection.execute(sql)
        results = cursor.fetchall()
        return {"success": True, "data": results, "count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}

@tool(APICallResult)
def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Make an API call"""
    try:
        import requests
        response = requests.request(method, endpoint, json=data)
        return {
            "status_code": response.status_code,
            "data": response.json() if response.content else None,
            "success": response.status_code < 400
        }
    except Exception as e:
        return {"status_code": 0, "data": None, "success": False, "error": str(e)}
```

### 2. Stateful Tools

Tools that maintain state across calls:

```python
class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.transaction_active = False
    
    @tool(ConnectionResult)
    def connect(self, connection_string: str):
        """Connect to database"""
        try:
            self.connection = create_connection(connection_string)
            return {"connected": True, "message": "Connected successfully"}
        except Exception as e:
            return {"connected": False, "message": str(e)}
    
    @tool(TransactionResult)
    def begin_transaction(self):
        """Begin a database transaction"""
        if not self.connection:
            return {"success": False, "message": "Not connected to database"}
        
        try:
            self.connection.begin()
            self.transaction_active = True
            return {"success": True, "message": "Transaction started"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    @tool(TransactionResult)
    def commit_transaction(self):
        """Commit the current transaction"""
        if not self.transaction_active:
            return {"success": False, "message": "No active transaction"}
        
        try:
            self.connection.commit()
            self.transaction_active = False
            return {"success": True, "message": "Transaction committed"}
        except Exception as e:
            return {"success": False, "message": str(e)}

# Usage
db_tools = DatabaseConnection()
agent = StrandsAgent(
    tools=[db_tools.connect, db_tools.begin_transaction, db_tools.commit_transaction],
    system_prompt=sop_prompt
)
```

### 3. Validation Tools

Tools that validate inputs and provide detailed feedback:

```python
@tool(ValidationResult)
def validate_email(email: str):
    """Validate an email address"""
    import re
    
    if not email:
        return {"valid": False, "message": "Email address is required"}
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return {"valid": True, "message": "Email address is valid", "email": email}
    else:
        return {"valid": False, "message": "Invalid email format"}

@tool(ValidationResult)
def validate_json(json_string: str):
    """Validate JSON format"""
    try:
        import json
        parsed = json.loads(json_string)
        return {"valid": True, "message": "Valid JSON", "data": parsed}
    except json.JSONDecodeError as e:
        return {"valid": False, "message": f"Invalid JSON: {str(e)}"}
```

## Integration Patterns

### 1. Tool-Driven State Machines

Design your procedures to respond to tool events:

```python
class ProcessDataProcedure(Procedure[DataState]):
    prompt = make_prompt("Process the uploaded data file")
    
    @handler
    def handle_file_read(self, state: DataState, event: FileReadEvent):
        path, content = event.payload
        
        if content is None:
            return f"Failed to read file {path}. Please check the file path and try again."
        
        # Store file content and move to validation
        state.raw_data = content
        state.file_path = path
        return ValidateDataProcedure()
    
    @handler
    def handle_validation_result(self, state: DataState, event: ValidationResult):
        if not event.payload["valid"]:
            return f"Data validation failed: {event.payload['message']}. Please fix the data and try again."
        
        # Data is valid, proceed to processing
        state.validated_data = event.payload["data"]
        return TransformDataProcedure()
```

### 2. Multi-Tool Workflows

Coordinate multiple tools in sequence:

```python
class ETLProcedure(Procedure[ETLState]):
    prompt = make_prompt("Extract, transform, and load data")
    
    @handler
    def handle_extraction_complete(self, state: ETLState, event: ExtractionResult):
        if not event.payload["success"]:
            return f"Data extraction failed: {event.payload['error']}. Please check the source."
        
        state.extracted_data = event.payload["data"]
        return TransformDataProcedure()
    
    @handler
    def handle_transformation_complete(self, state: ETLState, event: TransformationResult):
        if not event.payload["success"]:
            return f"Data transformation failed: {event.payload['error']}. Please check the transformation rules."
        
        state.transformed_data = event.payload["data"]
        return LoadDataProcedure()
    
    @handler
    def handle_load_complete(self, state: ETLState, event: LoadResult):
        if not event.payload["success"]:
            return f"Data loading failed: {event.payload['error']}. Please check the destination."
        
        state.records_loaded = event.payload["count"]
        return End()
```

### 3. Error Recovery with Tools

Use tools for error recovery and diagnostics:

```python
class ResilientProcedure(Procedure[MyState]):
    @handler
    def handle_operation_failed(self, state: MyState, event: OperationFailed):
        # Use diagnostic tool to understand the failure
        return DiagnosticProcedure()

class DiagnosticProcedure(Procedure[MyState]):
    prompt = make_prompt("Diagnose the system to understand the failure")
    
    @handler
    def handle_diagnostic_result(self, state: MyState, event: DiagnosticResult):
        diagnosis = event.payload
        
        if diagnosis["issue_type"] == "network":
            return RetryWithBackoffProcedure()
        elif diagnosis["issue_type"] == "permission":
            return RequestPermissionsProcedure()
        elif diagnosis["issue_type"] == "resource":
            return WaitForResourcesProcedure()
        else:
            return f"Unknown issue type: {diagnosis['issue_type']}. Manual intervention required."

@tool(DiagnosticResult)
def diagnose_system():
    """Diagnose system health and identify issues"""
    try:
        # Check network connectivity
        import requests
        requests.get("https://httpbin.org/get", timeout=5)
        network_ok = True
    except:
        network_ok = False
    
    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage("/")
    disk_ok = disk_usage.free > 1024 * 1024 * 100  # 100MB free
    
    # Determine issue type
    if not network_ok:
        issue_type = "network"
    elif not disk_ok:
        issue_type = "resource"
    else:
        issue_type = "unknown"
    
    return {
        "network_ok": network_ok,
        "disk_ok": disk_ok,
        "issue_type": issue_type,
        "timestamp": time.time()
    }
```

## File System Example Integration

The file system operator demonstrates sophisticated tool integration:

### Tool Definitions

```python
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

### SOP Integration

```python
def start_task(src_path: str, dst_path: str, backup_path: str):
    # Create SOP that observes tool events
    sop = FileSystemOperator(src_path, dst_path, backup_path).observes(
        eReadDirectory, eFileDeleteRequest, eFileDeleted, eCopyRequest, eFileCopied
    )
    
    # Generate prompt from state machine
    objective = "Migrate files from source to destination with backup safety"
    prompt = make_prompt_from_procedure(sop.init_proc, objective)
    
    # Create agent with tools
    agent = StrandsAgent(tools=[read_dir, cp, delete], system_prompt=prompt)
    
    # Add error recovery using the agent
    sop = sop.with_recovery(
        error_handler=lambda prompt: agent(f"Error occurred: {prompt}"),
        retry_limits=3
    )

    # Execute with tool integration
    with sop:
        agent(f"Move all files from {src_path} to {dst_path} with backup at {backup_path} using given tools only.")
```

## Next Steps

- [File System Example](08-file-system-example.md) - See tool integration in action
- [Advanced Patterns](09-advanced-patterns.md) - Complex integration scenarios
- [Error Recovery](06-error-recovery.md) - Recovery strategies with tools
