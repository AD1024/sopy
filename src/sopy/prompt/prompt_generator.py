import ast
import inspect
import textwrap
from typing import Dict, List, Set, Optional, Type, Tuple
from dataclasses import dataclass
from ..core.procedure import Procedure, Sigma, End
from ..core.event import Event
from . import make_prompt, Prompt
from collections import defaultdict
from functools import lru_cache


@dataclass
class HandlerInfo:
    """Information about an event handler method."""
    method_name: str
    event_type: Type[Event]
    returns_procedure: bool
    returns_string: bool
    docstring: Optional[str] = None


@dataclass
class ProcedureNode:
    """Represents a procedure in the graph."""
    procedure: Type[Procedure]
    name: str
    prompt_text: str
    handlers: Dict[Type[Event], HandlerInfo]
    is_terminal: bool = False
    docstring: Optional[str] = None

    @lru_cache
    def to_key(self):
        """Generate a unique key for this procedure node."""
        return (self.name, self.prompt_text)
    
    def __hash__(self):
        """Hash based on the unique key."""
        return hash((self.name, self.prompt_text))
    
    def __eq__(self, other):
        """Equality based on the unique key."""
        if not isinstance(other, ProcedureNode):
            return NotImplemented
        return self.procedure == other.procedure and self.prompt_text == other.prompt_text
    
    def __str__(self):
        """String representation of the procedure node."""
        return f"ProcedureNode(name={self.name}, prompt_text={self.prompt_text}, is_terminal={self.is_terminal})"


@dataclass
class ProcedureEdge:
    """Represents a transition between procedures."""
    from_procedure: ProcedureNode
    to_procedure: ProcedureNode
    event_type: Type[Event]
    condition: str
    is_error_recovery: bool = False


class ProcedureGraph:
    """Represents the complete procedure workflow as a graph."""
    
    def __init__(self):
        self.nodes: Dict[Tuple[str, str], ProcedureNode] = {}
        self.edges: Dict[Tuple[str, str], List[ProcedureEdge]] = defaultdict(list)
        self.entry_point: Optional[ProcedureNode] = None
        self.end_points: Set[str] = set()
    
    def add_node(self, node: ProcedureNode):
        """Add a procedure node to the graph."""
        self.nodes[node.name, node.prompt_text] = node
        if node.is_terminal:
            self.end_points.add(node.name)
    
    def add_edge(self, edge: ProcedureEdge):
        """Add a transition edge to the graph."""
        self.edges[edge.from_procedure.name, edge.from_procedure.prompt_text].append(edge)
    
    def get_successors(self, procedure_name: str, prompt: str) -> List[ProcedureNode]:
        """Get all procedures that can be reached from the given procedure."""
        return [edge.to_procedure for edge in self.edges.get((procedure_name, prompt), [])]


class ProcedureGraphBuilder:
    """Builds a procedure graph through reflection and analysis."""
    
    def __init__(self):
        self.discovered_procedures: Set[Tuple[Type[Procedure], str]] = set()
        self.graph = ProcedureGraph()
        self.context = None
    
    def build_graph(self, entry_procedure: Procedure[Sigma]) -> ProcedureGraph:
        """Build the complete procedure graph starting from the entry procedure."""
        self.context = inspect.getmodule(entry_procedure.__class__)
        self.graph.entry_point = self._create_procedure_node(entry_procedure.__class__, str(entry_procedure.prompt))
        self._discover_procedures(entry_procedure.__class__, str(entry_procedure.prompt))
        self._build_transitions()
        return self.graph
    
    def _discover_procedures(self, procedure: Type[Procedure], prompt_text: str):
        """Recursively discover all procedures reachable from the given procedure."""
        # Create procedure node
        node = self._create_procedure_node(procedure, prompt_text)
        print(node)
        if node.to_key() in self.graph.nodes:
            # If already discovered, skip
            return
        
        self.discovered_procedures.add((procedure, prompt_text))
        self.graph.add_node(node)
        
        # Discover connected procedures through handlers
        for handler_info in node.handlers.values():
            # Analyze handler to find returned procedure types
            returned_procedures = self._analyze_handler_returns(procedure, handler_info.method_name)
            for (proc_class, prompt) in returned_procedures:
                if (proc_class, prompt_text) not in self.discovered_procedures:
                    self._discover_procedures(proc_class, prompt)
    
    def _create_procedure_node(self, procedure: Type[Procedure], prompt: Optional[str] = None) -> ProcedureNode:
        """Create a procedure node by analyzing the procedure class."""
        name = procedure.__name__
        is_terminal = issubclass(procedure, End)
        
        # Extract prompt text
        prompt_text = ""
        if prompt:
            prompt_text = prompt
        elif hasattr(procedure, 'prompt') and procedure.prompt:
            prompt_text = str(procedure.prompt)
        
        # Extract docstring
        docstring = inspect.getdoc(procedure)
        
        # Find all handlers
        handlers = self._find_handlers(procedure)
        
        return ProcedureNode(
            procedure=procedure,
            name=name,
            prompt_text=prompt_text,
            handlers=handlers,
            is_terminal=is_terminal,
            docstring=docstring
        )
    
    def _find_handlers(self, procedure_class: Type[Procedure]) -> Dict[Type[Event], HandlerInfo]:
        """Find all event handlers in the procedure class."""
        handlers = {}
        
        for attr_name in dir(procedure_class):
            attr = getattr(procedure_class, attr_name)
            if callable(attr) and hasattr(attr, 'is_handler') and hasattr(attr, 'event_type'):
                event_type = getattr(attr, 'event_type')
                
                # Analyze return type hints
                sig = inspect.signature(attr)
                return_annotation = sig.return_annotation
                
                returns_procedure = False
                returns_string = False
                
                if return_annotation != inspect.Signature.empty:
                    if hasattr(return_annotation, '__origin__'):  # Union type
                        args = getattr(return_annotation, '__args__', ())
                        returns_string = str in args or any(
                            isinstance(arg, type) and issubclass(arg, str) 
                            for arg in args 
                        )
                        returns_procedure = any(
                            isinstance(arg, type) and issubclass(arg, Procedure) 
                            for arg in args
                        )
                    elif isinstance(return_annotation, type):
                        try:
                            returns_string = issubclass(return_annotation, str)
                        except TypeError:
                            returns_string = False
                        try:
                            returns_procedure = issubclass(return_annotation, Procedure)
                        except TypeError:
                            returns_procedure = False
                
                handlers[event_type] = HandlerInfo(
                    method_name=attr_name,
                    event_type=event_type,
                    returns_procedure=returns_procedure,
                    returns_string=returns_string,
                    docstring=inspect.getdoc(attr)
                )
        
        return handlers
    
    def _analyze_handler_returns(self, procedure: Type[Procedure], method_name: str) -> Set[Tuple[Type[Procedure], str]]:
        """Analyze a handler method to find what procedure types it can return."""
        returned_procedures = set()
        
        try:
            method = getattr(procedure, method_name)
            (source, _lineno) = inspect.getsourcelines(method)
            src = [textwrap.fill(src, tabsize=4, width=9999) for src in source]
            
            # Parse the AST to find return statements
            tree = ast.parse(textwrap.dedent('\n'.join(src)))
            print("Analyzing handler:", procedure.__name__, method_name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return) and node.value:
                    returned_procedures.update(self._extract_procedure_types_from_ast(node.value))
        
        except (OSError, TypeError):
            # If source is not available, skip AST analysis
            pass
        
        return returned_procedures
    
    def _extract_procedure_types_from_ast(self, node: ast.AST) -> Set[Tuple[Type[Procedure], str]]:
        """Extract procedure types from AST nodes."""
        procedure_types = set()

        def _try_extract_prompt_from_call(node: ast.expr) -> Optional[str]:
            if not isinstance(node, ast.Call):
                return None
            if isinstance(node.func, ast.Name) and node.func.id == 'make_prompt':
                if node.args:
                    # Extract the first argument as the prompt text
                    return node.args[0].s if isinstance(node.args[0], ast.Constant) else None
            return None
        
        if isinstance(node, ast.Call):
            # prompt may be passed as an argument
            # if so, extract the prompt text
            prompt_txt = None
            if node.args or node.keywords:
                prompt_arg = [x
                              for x in filter(lambda x: x,
                                [_try_extract_prompt_from_call(n)
                                    for n in node.args + [v.value for v in node.keywords]])]
                if len(prompt_arg) == 1:
                    prompt_txt = prompt_arg[0]
            # Constructor call like Copy() or End()
            if isinstance(node.func, ast.Name):
                class_name = node.func.id
                # Try to resolve the class name to an actual class
                proc_class = getattr(self.context, class_name, None)
                if not proc_class or not issubclass(proc_class, Procedure):
                    return procedure_types
                # Add the procedure class and its prompt text
                procedure_types.add((proc_class, prompt_txt if prompt_txt else str(proc_class.prompt)))
        
        elif isinstance(node, ast.Name):
            # Direct class reference
            proc_class = getattr(self.context, node.id, None)
            if proc_class and issubclass(proc_class, Procedure):
                # If it's a procedure class, add it with its default prompt
                procedure_types.add((proc_class, str(proc_class.prompt)))
        
        return procedure_types
    
    def _build_transitions(self):
        """Build transition edges between procedures."""
        for (name, prompt), node in self.graph.nodes.items():
            for event_type, handler_info in node.handlers.items():
                # Find procedures returned by this handler
                returned_procedures = self._analyze_handler_returns(node.procedure, handler_info.method_name)
                
                for (target_procedure, prompt) in returned_procedures:
                    target_name = target_procedure.__name__
                    if (target_name, prompt) not in self.graph.nodes:
                        # If the target procedure is not yet in the graph, create it
                        assert False, f"Target procedure {target_name}, {prompt} not found in the graph. Ensure it is defined and reachable."
                        target_node = self._create_procedure_node(target_procedure, prompt)
                        self.graph.add_node(target_node)
                    else:
                        target_node = self.graph.nodes[target_name, prompt]
                    
                    # Create edge
                    condition = f"When {event_type.__name__} occurs"
                    is_error_recovery = handler_info.returns_string
                    
                    edge = ProcedureEdge(
                        from_procedure=node,
                        to_procedure=target_node,
                        event_type=event_type,
                        condition=condition,
                        is_error_recovery=is_error_recovery
                    )
                    
                    self.graph.add_edge(edge)

class PromptBuilder:
    """Builds SOP prompts from procedure graphs."""
    
    def __init__(self, format_style: str = "natural_language"):
        self.format_style = format_style
    
    def build_sop_prompt(self, graph: ProcedureGraph, objective: str = "") -> str:
        """Build a complete SOP prompt from the procedure graph."""
        if self.format_style == "natural_language":
            return self._build_natural_language_prompt(graph, objective)
        else:
            raise ValueError(f"Unsupported format style: {self.format_style}")
    
    def _build_natural_language_prompt(self, graph: ProcedureGraph, objective: str) -> str:
        """Build a natural language SOP prompt."""
        sections = []
        
        # Header
        if objective:
            sections.append(f"STANDARD OPERATING PROCEDURE\n\nOBJECTIVE: {objective}\n")
        else:
            sections.append("STANDARD OPERATING PROCEDURE\n")
        
        # Main procedure flow
        sections.append("MAIN PROCEDURE:")
        main_flow = self._build_main_flow(graph)
        sections.append(main_flow)
        
        # Error handling
        error_handling = self._build_error_handling(graph)
        if error_handling:
            sections.append("\nERROR HANDLING:")
            sections.append(error_handling)
        
        # Completion criteria
        completion = self._build_completion_criteria(graph)
        if completion:
            sections.append("\nCOMPLETION:")
            sections.append(completion)
        
        return "\n".join(sections)
    
    def _build_main_flow(self, graph: ProcedureGraph) -> str:
        """Build the main procedure flow description."""
        if not graph.entry_point:
            return "No entry point defined."
        
        flow_steps = []
        visited = set()
        step_number = 1
        
        # Perform a traversal to build the main flow
        def traverse_main_path(node: ProcedureNode, depth: int = 0):
            nonlocal step_number
            key = node.to_key()
            
            if key in visited or key not in graph.nodes:
                return
            
            visited.add(key)
            
            # Build step description
            step_desc = f"{step_number}. {node.name.upper()}: {node.prompt_text}"
            
            # Add transition information
            successors = graph.get_successors(*key)
            non_error_successors = [
                succ for succ in successors 
                if not any(edge.is_error_recovery for edge in graph.edges[key] 
                          if edge.from_procedure == node and edge.to_procedure == succ)
            ]
            
            if non_error_successors:
                # Find the events that lead to successors
                events_to_successors = {}
                for edge in graph.edges[key]:
                    if edge.from_procedure == node and edge.to_procedure in non_error_successors:
                        events_to_successors[edge.to_procedure] = edge.event_type.__name__
                
                if len(non_error_successors) == 1:
                    next_proc = non_error_successors[0]
                    event_name = events_to_successors.get(next_proc, "completion")
                    step_desc += f" Wait for {event_name} event, then proceed to {next_proc.name}."
                else:
                    step_desc += " Depending on the event received, proceed to the appropriate next step."
            
            flow_steps.append(step_desc)
            step_number += 1
            
            # Continue with the main path (first non-error successor)
            if non_error_successors:
                traverse_main_path(non_error_successors[0], depth + 1)
        
        traverse_main_path(graph.entry_point)
        
        return "\n".join(flow_steps)
    
    def _build_error_handling(self, graph: ProcedureGraph) -> str:
        """Build error handling descriptions."""
        error_cases = []
        
        for edges in graph.edges.values():
            for edge in edges:
                if edge.is_error_recovery:
                    error_desc = f"- If {edge.event_type.__name__} fails or causes errors: {edge.condition}"
                    error_cases.append(error_desc)
        
        # Add general error handling
        error_cases.append("- If unrecognized events occur: Request clarification and continue with current procedure")
        error_cases.append("- If procedure violations are detected: Provide specific recovery instructions")
        
        return "\n".join(error_cases)
    
    def _build_completion_criteria(self, graph: ProcedureGraph) -> str:
        """Build completion criteria description."""
        if not graph.end_points:
            return "The procedure completes when no further transitions are possible."
        
        end_descriptions = []
        for end_point in graph.end_points:
            if end_point in graph.nodes:
                node = graph.nodes[end_point]
                if hasattr(node.procedure, 'prompt'):
                    # For End procedures with action descriptions
                    end_descriptions.append(f"- {end_point}: {node.prompt_text}")
                else:
                    end_descriptions.append(f"- {end_point}: Procedures complete with no further actions required.")
        
        if end_descriptions:
            return "The procedure completes successfully when:\n" + "\n".join(end_descriptions)
        else:
            return "The procedure completes when the terminal state is reached."


def make_prompt_from_procedure(p: Procedure[Sigma], objective: str = "", format_style: str = "natural_language") -> Prompt:
    """
    Generate an SOP prompt from a procedure by building and traversing the procedure graph.
    
    Args:
        p: The entry procedure to start building the graph from
        objective: Optional objective description for the SOP
        format_style: The format style for the prompt (default: "natural_language")
    
    Returns:
        Prompt: A prompt containing the complete SOP description
    """
    # Build the procedure graph
    builder = ProcedureGraphBuilder()
    graph = builder.build_graph(p)
    print(graph.edges.keys(), graph.nodes.keys())
    
    # Generate the prompt
    prompt_builder = PromptBuilder(format_style)
    prompt_text = prompt_builder.build_sop_prompt(graph, objective)
    
    return make_prompt(prompt_text)
