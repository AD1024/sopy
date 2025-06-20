# Support for Strands agents events
import inspect
import traceback
from ast import literal_eval
from typing import Any, Dict, List, Optional, Callable, Iterable, TypeVar
from types import get_original_bases
from functools import wraps
from strands import Agent
from strands.types.tools import ToolResult, ToolUse, ToolResultContent
from strands.tools import tool as strands_tool
from strands.tools.decorator import FunctionToolMetadata
from strands.types.content import Message, ContentBlock, Role
from ..core import Event, EventBus
from ..core.sop import SOPRetryExceptionHook, SOPViolation
from ..utils import Log


# Event classes for different interaction types
class eHumanMessage(Event[Any]):
    """Event representing a message from a human user to the AI agent."""
    pass


class eAIResponse(Event[Any]):
    """Event representing a response from the AI agent to a human user."""
    pass


class eToolUse(Event[ToolUse]):
    """Event representing an AI agent requesting to use a tool."""
    pass


class eToolResult(Event[ToolResult]):
    """Event representing the result returned from a tool execution."""
    pass


class eA2A(Event[Any]):
    """Event representing communication between AI agents in multi-agent scenarios."""
    pass


class eContentBlock(Event[ContentBlock]):
    """Event representing a generic content block from Strands messages."""
    pass

class ToolDefException(Exception):
    """Exception raised when a tool definition is invalid."""
    def __init__(self, message: str):
        super().__init__(message)


PayloadT = TypeVar('PayloadT', bound=Any)

def _eval_literal(value: str) -> Any:
    try:
        return literal_eval(value)
    except SyntaxError:
        return str(value)

def tool(result_event: Callable[..., Event[PayloadT]] = eToolResult, use_event: Optional[Callable[..., Event]] = None) -> Callable[[Callable[..., PayloadT]], Callable[..., PayloadT]]:
    def decorator(func: Callable[..., PayloadT]):
        if use_event is not None:
            # check the constructor accepts a `toolUseId` and all parameters of `func`
            ctor_meta = FunctionToolMetadata(use_event.__init__)
            func_meta = FunctionToolMetadata(func)
            if not "toolUseId" in ctor_meta.signature.parameters:
                raise ToolDefException(f"Tool use event {use_event.__name__} must accept a 'toolUseId' parameter.")
            if not all(param in ctor_meta.signature.parameters for param in func_meta.signature.parameters):
                missing_params = [param for param in func_meta.signature.parameters if param not in ctor_meta.signature.parameters]
                raise ToolDefException(f"Tool use event {use_event.__name__} must accept all parameters of the tool function {func.__name__}: {', '.join(missing_params)}.")

        decorated = strands_tool(func)

        @wraps(decorated)
        def wrapped(*args, **kwargs):
            try:
                if not (
                    len(args) > 0
                    and isinstance(args[0], dict)
                    and (not args[0] or "toolUseId" in args[0] or "input" in args[0])
                ):
                    # taken from strands: this means it is not a tool use, perform reguar function call
                    return func(*args, **kwargs)
                if use_event is None:
                    tool_use = eToolUse(
                        payload=ToolUse(toolUseId=args[0]['toolUseId'], name=args[0]['name'], input=args[0].get('input', None)),
                        sender=kwargs.get('agent', None),
                        target=None)
                else:
                    tool_use = use_event(toolUseId=args[0]['toolUseId'], **args[0].get('input', {}))
                try:
                    EventBus.push_event(tool_use)
                    result = decorated(*args, **kwargs)
                    
                    arranged_result = {
                        'toolUseId': result['toolUseId'],
                        'payload': _eval_literal(list(result.get('content', [None])[0].values())[0]),
                        'target': kwargs.get('agent', None),
                        'description': None,
                    }
                    base = get_original_bases(result_event)

                    # get payload type
                    if base and hasattr(base[0], '__args__') and base[0].__args__:
                        payload_type = base[0].__args__[0]
                    else:
                        Log.warning(f"Tool result event {result_event.__name__} does not specify a payload type, using Any.")
                        payload_type = Any

                    # re-init input model with the payload type
                    ctor_meta = FunctionToolMetadata(result_event.__init__)
                    ctor_meta.type_hints['payload'] = payload_type
                    ctor_meta.input_model = ctor_meta._create_input_model()

                    validated_result = ctor_meta.validate_input(arranged_result)
                    e = result_event(**validated_result)
                    EventBus.push_event(e)
                except SOPRetryExceptionHook as e:
                    Log.warning(f"Retrying with recovery prompt {e.prompt}")
                    return {
                            "toolUseId": args[0]['toolUseId'],
                            "status": "error",
                            "content": [{"text": f"An error occured in Procedure {e.current_procedure.__class__.__name__}: {str(e.prompt)}. Prompt for this procedure is {e.current_procedure.prompt}"}],
                    }
                except SOPViolation as e:
                    Log.error(f"SOP Violation: {e}")
                    return {
                        "toolUseId": args[0]['toolUseId'],
                        "status": "error",
                        "content": [{"text": f"SOP Violation: {str(e)}"}],
                    }
            except Exception as e:
                Log.error(f"Error while processing tool result: {e} ({type(e).__name__}))")
                traceback.print_exc()
                return exit(1)
            
            return result

        return wrapped
    return decorator
    
class StrandsAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        return result