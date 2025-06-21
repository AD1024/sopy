# Getting Started with SOPy

Welcome to SOPy! This tutorial will introduce you to the core concepts and help you build your first Standard Operating Procedure (SOP) for LLM agents.

## What is SOPy?

SOPy is a framework that helps you create reliable, structured workflows for AI agents by:

1. **Defining workflows as state machines** - Each step is a clear, well-defined state
2. **Automatically generating prompts** - Your state machine becomes natural language instructions
3. **Handling errors gracefully** - Built-in recovery mechanisms prevent agents from getting stuck
4. **Monitoring agent behavior** - Event-driven architecture provides full observability

## Core Components

### 1. Procedures (States)
A `Procedure` represents a state in your workflow. It defines what the agent should do and how to respond to events.

```python
from sopy import Procedure, make_prompt

class GreetUser(Procedure[dict]):
    prompt = make_prompt("Greet the user warmly and ask how you can help")
```

### 2. Events
An `Event` represents something that happened - like a tool being called or a task being completed.

```python
from sopy import Event

class UserResponded(Event[str]):
    """Event triggered when user provides input"""
    pass
```

### 3. Handlers
Handlers are methods that process events and determine what happens next.

```python
from sopy import handler

class GreetUser(Procedure[dict]):
    prompt = make_prompt("Greet the user warmly and ask how you can help")
    
    @handler
    def handle_user_responded(self, state: dict, event: UserResponded):
        # Process the user's response
        user_message = event.payload
        state['last_message'] = user_message
        
        # Transition to next procedure
        return ProcessRequest()
```

### 4. SOP (Standard Operating Procedure)
The `SOP` class orchestrates everything - it manages state, handles events, and coordinates transitions.

```python
from sopy import SOP

# Create the SOP
sop = SOP(
    state={'messages': []},  # Initial state
    init_proc=GreetUser()    # Starting procedure
).observes(UserResponded)    # Events to listen for
```

## Your First SOP: Simple Chat Bot

Let's build a simple chatbot that greets users and responds to their messages.

### Step 1: Define Your State

```python
from pydantic import BaseModel, Field
from typing import List

class ChatState(BaseModel):
    messages: List[str] = Field(default_factory=list, description="Chat message history")
    user_name: str = Field(default="", description="User's name if provided")
    conversation_active: bool = Field(default=True, description="Whether conversation is ongoing")
```

### Step 2: Define Events

```python
from sopy import Event

class UserMessage(Event[str]):
    """User sent a message"""
    pass

class UserWantsToExit(Event[bool]):
    """User wants to end the conversation"""
    pass
```

### Step 3: Create Procedures

```python
from sopy import Procedure, handler, make_prompt, End

class Greeting(Procedure[ChatState]):
    prompt = make_prompt("Greet the user and ask for their name")
    
    @handler
    def handle_user_message(self, state: ChatState, event: UserMessage):
        message = event.payload.lower()
        
        # Check if user provided their name
        if "my name is" in message or "i'm" in message or "i am" in message:
            # Extract name (simplified)
            words = message.split()
            if "is" in words:
                name_idx = words.index("is") + 1
                if name_idx < len(words):
                    state.user_name = words[name_idx].capitalize()
            
            return Chatting()
        
        # Ask for name again if not provided
        return "I'd love to know your name! Please tell me what I should call you."

class Chatting(Procedure[ChatState]):
    prompt = make_prompt("Have a friendly conversation with the user")
    
    @handler
    def handle_user_message(self, state: ChatState, event: UserMessage):
        message = event.payload
        state.messages.append(f"User: {message}")
        
        # Check for exit conditions
        if any(word in message.lower() for word in ["bye", "goodbye", "exit", "quit"]):
            return Farewell()
        
        # Continue conversation
        return self
    
    @handler
    def handle_user_wants_to_exit(self, state: ChatState, event: UserWantsToExit):
        return Farewell()

class Farewell(Procedure[ChatState]):
    prompt = make_prompt("Say goodbye to the user warmly")
    
    @handler
    def handle_user_message(self, state: ChatState, event: UserMessage):
        # End the conversation
        state.conversation_active = False
        return End()
```

### Step 4: Create and Run the SOP

```python
def run_chatbot():
    # Create the SOP
    sop = SOP(
        state=ChatState(),
        init_proc=Greeting()
    ).observes(UserMessage, UserWantsToExit)
    
    # Add error recovery
    def handle_error(prompt):
        print(f"Error occurred: {prompt}")
        return "I'm sorry, something went wrong. Let me try again."
    
    sop = sop.with_recovery(
        error_handler=handle_error,
        retry_limits=3
    )
    
    # Run the SOP
    with sop:
        print("Chatbot started! Type 'quit' to exit.")
        
        while sop.state.conversation_active:
            user_input = input("> ")
            
            # Create and push events
            if user_input.lower() in ['quit', 'exit']:
                from sopy.core.event import EventBus
                EventBus.push_event(UserWantsToExit(payload=True))
            else:
                from sopy.core.event import EventBus
                EventBus.push_event(UserMessage(payload=user_input))

if __name__ == "__main__":
    run_chatbot()
```

## Understanding the Flow

1. **Initialization**: SOP starts in `Greeting` state
2. **Event Handling**: When user types, `UserMessage` event is created
3. **State Transition**: Handler processes event and returns next procedure
4. **Error Recovery**: If something goes wrong, error handler provides recovery prompt
5. **Completion**: When `End()` is returned, the SOP completes successfully

## Key Benefits

- **Predictable Behavior**: State machine ensures agent follows defined workflow
- **Error Recovery**: Built-in mechanisms handle unexpected situations
- **Type Safety**: Full type checking prevents runtime errors
- **Observability**: Event system provides complete audit trail
- **Testability**: Each procedure can be tested independently

## Next Steps

Now that you understand the basics, explore:

- [Procedures and Events](02-procedures-and-events.md) - Deep dive into core components
- [State Management](03-state-management.md) - Advanced state handling patterns
- [File System Example](08-file-system-example.md) - Real-world complex example

## Common Patterns

### Conditional Transitions
```python
@handler
def handle_event(self, state: MyState, event: MyEvent):
    if state.some_condition:
        return NextProcedure()
    else:
        return "Condition not met, please try again"
```

### Self-Loops (Staying in Same State)
```python
@handler
def handle_event(self, state: MyState, event: MyEvent):
    # Process event but stay in same state
    state.counter += 1
    return self
```

### Error Recovery
```python
@handler
def handle_event(self, state: MyState, event: MyEvent):
    if not event.payload:
        return "Operation failed, please retry"
    return NextProcedure()
```

This foundation will serve you well as you build more complex SOPs. The key is to think in terms of states, events, and transitions - this makes your agent behavior predictable and reliable.
