# SOPy: Structural Standard Operating Procedure Framework

SOPy is a state machine-based framework for monitoring and controlling LLM agent actions with automatic recovery mechanisms. It enables developers to create reliable, structured workflows for AI agents by encoding Standard Operating Procedures (SOPs) as state machines that automatically generate natural language prompts and handle error recovery.

## 🚀 Key Features

- **State Machine-Based Architecture**: Define agent workflows as clear, structured state machines
- **Automatic Prompt Generation**: Convert state machines into natural language SOP prompts automatically
- **Built-in Error Recovery**: Automatic retry mechanisms with customizable error handling
- **Event-Driven Design**: Actions trigger events that cause predictable state transitions
- **Type Safety**: Full type annotations with generic state management
- **Tool Integration**: Seamless integration with Strands agents and external tools
- **Monitoring & Logging**: Comprehensive logging and event tracing capabilities

## 🎯 Why SOPy?

Traditional LLM agent frameworks often lack structure and reliability. SOPy addresses these challenges by:

1. **Providing Structure**: State machines make agent behavior predictable and debuggable
2. **Ensuring Reliability**: Built-in error recovery prevents agents from getting stuck
3. **Enabling Verification**: Generated prompts can be cross-checked and validated
4. **Simplifying Development**: High-level abstractions reduce boilerplate code
5. **Supporting Monitoring**: Event-driven architecture enables comprehensive observability

## 🏃 Quick Start

Here's a simple example showing how to create a basic SOP:

```python
from sopy import Procedure, Event, SOP, handler, make_prompt, End

# Define your state
class TaskState:
    def __init__(self):
        self.completed = False

# Define events
class TaskCompleted(Event[bool]): pass

# Define procedures (states)
class StartTask(Procedure[TaskState]):
    prompt = make_prompt("Begin the task and work towards completion")
    
    @handler
    def handle_task_completed(self, state: TaskState, event: TaskCompleted):
        if event.payload:  # Task was successful
            state.completed = True
            return End()
        else:
            return "Task failed, please try again"

# Create and run the SOP
def run_example():
    sop = SOP(
        state=TaskState(),
        init_proc=StartTask()
    ).observes(TaskCompleted)
    
    with sop:
        # Your agent logic here
        pass
```

## 🏗️ Core Concepts

### Procedures
Procedures represent states in your SOP state machine. Each procedure defines:
- A **prompt**: Natural language description of what should be done
- **Event handlers**: Methods that process events and determine state transitions
- **Local state**: Procedure-specific variables and logic

### Events
Events are messages that trigger state transitions. They carry:
- **Payload**: Data associated with the event
- **Metadata**: Sender, target, description, and tool use information
- **Type safety**: Generic typing ensures payload type correctness

### Handlers
Handlers are methods that process events and can return:
- **Next Procedure**: Transition to a new state
- **Self**: Stay in the current state
- **String**: Error message triggering recovery mechanism

### State Management
SOPy uses a generic state pattern where `Procedure[T]` operates on state of type `T`. This enables:
- Type-safe state access across procedures
- Shared state between different procedure instances
- Clear data flow and state evolution tracking

## 📚 Documentation

For detailed tutorials and examples, see the `tutorial/` directory:

- [Getting Started](tutorial/01-getting-started.md) - Basic concepts and simple examples
- [Procedures and Events](tutorial/02-procedures-and-events.md) - Deep dive into core components
- [State Management](tutorial/03-state-management.md) - Managing state with the Sigma pattern
- [Handlers and Transitions](tutorial/04-handlers-and-transitions.md) - Event handling patterns
- [Automatic Prompts](tutorial/05-automatic-prompts.md) - Prompt generation system
- [Error Recovery](tutorial/06-error-recovery.md) - Recovery mechanisms and retry logic
- [Tool Integration](tutorial/07-tool-integration.md) - Working with Strands agents
- [File System Example](tutorial/08-file-system-example.md) - Complete real-world example
- [Advanced Patterns](tutorial/09-advanced-patterns.md) - Complex SOPs and best practices

## 🔧 Example: File System Operator

The `examples/file_system_operator/` directory contains a complete example that demonstrates:

- **Complex State Management**: Tracking multiple file sets during migration
- **Multi-Step Procedures**: ReadDir → BackUpCopy → MigrateCopy → Clear
- **Error Handling**: Automatic rollback on failure (Abort procedure)
- **Tool Integration**: File system operations through Strands tools
- **Automatic Prompts**: Generated SOP from state machine definition

```python
# The file system operator automatically generates this SOP:
"""
Procedure ReadDir:
    Instruction: Read the directory to be migrated
    Action: eReadDirectory

Procedure BackUpCopy:
    Instruction: For each file in the directory, copy it to the given backup location.
    If retry limit is reached, then go to Abort.
    If all files copied to backup location, then go to MigrateCopy.
    Action: eFileCopied, eCopyRequest

Procedure MigrateCopy:
    Instruction: For each file in the directory, copy it to the migrate location.
    If retry limit is reached, then go to Abort.
    If all files copied to the migration location, then go to Clear.
    Action: eFileCopied, eCopyRequest

# ... and more
"""
```

## 🛠️ Development Status

**This project is under active development.** Current features include:

- ✅ Core state machine framework
- ✅ Event system and handlers
- ✅ Automatic prompt generation
- ✅ Error recovery mechanisms
- ✅ Strands agent integration
- ✅ File system operator example
- 🚧 Additional integrations (LangChain, etc.)
- 🚧 Advanced monitoring, debugging and tracing tools
- 🚧 Performance optimizations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to:

- Report bugs and request features
- Submit pull requests
- Improve documentation
- Share examples and use cases

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🔗 Related Projects

- [Strands](https://github.com/strands-ai/strands) - LLM agent framework with tool integration
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing LLM applications

---

For more information, examples, and detailed documentation, explore the `tutorial/` directory and the `examples/` folder.
