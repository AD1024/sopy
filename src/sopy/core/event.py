from typing import Optional, Any, Generic, TypeVar, Callable, List, Iterable


T = TypeVar('T', bound=Any)
class Event(Generic[T]):
    """
    The base class of an event
    """
    description: Optional[str]
    sender: Any
    target: Any
    payload: T
    toolUseId: Optional[str]

    def __init__(self, payload: T, description: Optional[str] = None, sender: Any = None, target: Any = None, toolUseId: Optional[str] = None):
        """
        Initialize the event with a payload, description, sender, and target.
        
        Args:
            payload: The data associated with the event.
            description: Optional description of the event.
            sender: The entity that sent the event.
            target: The entity that is the intended recipient of the event.
        """
        self.payload = payload
        self.description = description
        self.sender = sender
        self.target = target
        self.toolUseId = toolUseId
    
    def __repr__(self) -> str:
        return show(self)
    
    def __str__(self) -> str:
        return self.__repr__()

class EventTranslator:

    translator_map: dict[type, Callable[[Any], Event[Any] | Iterable[Event[Any]]]]

    @staticmethod
    def register():
        """Decorator to register a function as an event translator."""
        def decorator(func: Callable[[Any], Event[Any] | Iterable[Event[Any]]]) -> Callable[[Any], Event[Any] | Iterable[Event[Any]]]:
            assert len(func.__annotations__) == 1, "Translator function must have exactly one type annotation."
            event_type = next(iter(func.__annotations__.values()))
            EventTranslator.translator_map[event_type] = func
            return func
        return decorator
    
    @staticmethod
    def translate(event: Any) -> Optional[Event[Any] | Iterable[Event[Any]]]:
        translator = EventTranslator.translator_map.get(type(event))
        if translator:
            return translator(event)
        return None

class EventBus:
    subscribers: dict[str, Callable[[Event], Any]] = {}
    func_to_uid: dict[Callable[[Event], Any], str] = {}
    trace: List[Event] = []

    @staticmethod
    def subscribe(uid: str, listener: Callable[[Event], Any]):
        """
        Subscribe a listener to the event bus with a unique identifier.
        
        Args:
            uid: Unique identifier for the listener.
            listener: The function to call when an event is published.
        """
        if uid in EventBus.subscribers:
            raise ValueError(f"Listener with UID '{uid}' already exists.")
        EventBus.subscribers[uid] = listener
    
    @staticmethod
    def unsubscribe(uid: str):
        """
        Unsubscribe a listener from the event bus using its unique identifier.
        
        Args:
            uid: Unique identifier for the listener to remove.
        """
        if uid not in EventBus.subscribers:
            raise ValueError(f"Listener with UID '{uid}' does not exist.")
        del EventBus.subscribers[uid]
    
    @staticmethod
    def push_event(event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish.
        """
        EventBus.trace.append(event)
        for listener in EventBus.subscribers.values():
            listener(event)

    @staticmethod
    def get_trace() -> List[Event]:
        """
        Get the trace of all events that have been published.
        
        Returns:
            A list of events in the order they were published.
        """
        return EventBus.trace

def subscribe(func):
    """
    Decorator to subscribe a function to the event bus.
    The function must accept an Event as its only argument.
    """
    uid = func.__name__
    if uid in EventBus.subscribers:
        raise ValueError(f"Listener with UID '{uid}' already exists.")
    
    EventBus.subscribers[uid] = func
    EventBus.func_to_uid[func] = uid
    
    return func

def show(event: Event) -> str:
    """
    Show an event
    """
    return f"{event.__class__.__name__}(sender={event.sender}, target={event.target}, payload={event.payload})"