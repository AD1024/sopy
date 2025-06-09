import ast
import inspect
import textwrap
import types
import traceback
from typing import Dict, List, Set, Optional, Type, Tuple, Self, Generic
from dataclasses import dataclass
from ..core.procedure import Procedure, Sigma, End
from ..core.event import Event
from . import make_prompt, Prompt
from collections import defaultdict
from functools import lru_cache
from ..utils import Log


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
    condition: List[ast.expr]
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
    
    def has_node(self, node: ProcedureNode) -> bool:
        """Check if the graph contains the given procedure node."""
        return node.to_key() in self.nodes


class ProcedureGraphBuilder(Generic[Sigma]):
    """Builds a procedure graph through reflection and analysis."""
    
    def __init__(self):
        self.discovered_procedures: Set[Tuple[Type[Procedure], str]] = set()
        self.graph = ProcedureGraph()
        self.context = None
    
    def build_graph(self, entry_procedure: Procedure[Sigma]) -> ProcedureGraph:
        """Build the complete procedure graph starting from the entry procedure."""
        self.context = inspect.getmodule(entry_procedure.__class__)
        if not self.context:
            raise ValueError("Entry procedure must be defined in a module.")
        analysis = HandlerAnalysis(self.context)
        analysis.init()
        return analysis.construct_procedure_graph(entry_procedure)

class HandlerAnalysis(ast.NodeVisitor):
    """Analyzes a procedure hanlder and computes transitions
       to the next procedure and their conditions (e.g., under an `if` statement).
    """
    class AnalysisScope:
        def __init__(self, parent: Optional[Self] = None, path_condition: Optional[ast.expr] = None, escape: bool = False):
            self.parent: Optional[Self] = parent
            # map from expr hash to a tuple of (value, condition of assignment)
            self.variables: Dict[int, ast.expr] = {}
            self.path_condition = path_condition
            self.escape = escape

        def _name_hash(self, name: ast.expr) -> int:
            if isinstance(name, ast.Name):
                return hash(name.id)
            elif isinstance(name, ast.Attribute):
                return hash((name.attr, self._name_hash(name.value)))
            elif isinstance(name, ast.Subscript):
                return hash((self._name_hash(name.value), self._name_hash(name.slice)))
            elif isinstance(name, ast.List):
                return hash(tuple(self._name_hash(e) for e in name.elts))
            elif isinstance(name, ast.Tuple):
                return hash(tuple(self._name_hash(e) for e in name.elts))
            elif isinstance(name, ast.Constant):
                return hash(name.value)
            raise TypeError(f"Unsupported name type for hashing during analysis: {type(name)}")
    
        def set_var(self, name: ast.expr, value: ast.expr):
            name_hash = self._name_hash(name)
            self.variables[name_hash] = value
        
        def get_var(self, name: ast.expr) -> Optional[ast.expr]:
            """Get a variable value by its name."""
            name_hash = self._name_hash(name)
            return self.variables.get(name_hash, None)

        def update(self, name: ast.expr, value):
            self.set_var(name, value)
            if self.parent and self.escape:
                self.parent.update(name, value)
        
        def assign(self, name: ast.expr, value: ast.expr):
            if self.parent:
                current_value = self.parent.look_up(name)
                if current_value is None:
                    self.update(name, value)
                else:
                    # The variable is defined in a parent scope, update it with IfExpr
                    new_value = ast.IfExp(test=ast.List(self.current_guard(), ctx=ast.Load()),
                                          body=value, 
                                          orelse=current_value)
                    self.update(name, new_value)
            else:
                self.update(name, value)

        
        def look_up(self, name: ast.expr) -> Optional[ast.expr]:
            """Look up a variable in the current scope or parent scopes."""
            name_hash = self._name_hash(name)
            if name_hash in self.variables:
                return self.variables[name_hash]
            if self.parent:
                return self.parent.look_up(name)
            return None
        
        @lru_cache
        def current_guard(self) -> List[ast.expr]:
            if self.parent is None:
                if self.path_condition:
                    return [self.path_condition]
                return []
            else:
                parent_guard = self.parent.current_guard()
                if self.path_condition:
                    return parent_guard + [self.path_condition]
                return parent_guard
        
        def push(self, path_condition: Optional[ast.expr] = None, escape: bool = True) -> 'HandlerAnalysis.AnalysisScope':
            """Push a new scope with an optional path condition."""
            return HandlerAnalysis.AnalysisScope(parent=self, path_condition=path_condition, escape=escape)
        
        def pop(self) -> 'HandlerAnalysis.AnalysisScope':
            """Pop the current scope and return the parent scope."""
            if self.parent is None:
                raise RuntimeError("Cannot pop from the root scope.")

            return self.parent
            

    def __init__(self, ctx: types.ModuleType):
        super().__init__()
        self.scope: HandlerAnalysis.AnalysisScope = HandlerAnalysis.AnalysisScope()
        self.cond = None
        self.escapes = [True]
        self.ctx = ctx
        self.current_node: Optional[ProcedureNode] = None
        self.current_handler: Optional[HandlerInfo] = None
        self.next_neighbor: List[ProcedureNode] = []
        self.graph = ProcedureGraph()
    
    def init(self):
        """Initialize the analysis with a new scope."""
        self.scope = HandlerAnalysis.AnalysisScope()
        self.cond = None
        self.current_node = None
        self.current_handler = None
        self.next_neighbor = []
    
    def path(self, cond: Optional[ast.expr] = None) -> Self:
        self.cond = cond
        return self

    def __enter__(self):
        self.scope = self.scope.push(self.cond, escape=self.escapes[-1])
        return self
    
    def __exit__(self, exc_type, exc_value, _tb):
        if exc_type is not None:
            Log.warning(f"Exception occurred during handler analysis: {exc_value}.\nStack trace: {traceback.format_exc()}")
        self.scope = self.scope.pop()
        self.cond = None
    
    def current_guard(self) -> List[ast.expr]:
        """Get the current path condition as a list of expressions."""
        return self.scope.current_guard()

    def assign(self, name: ast.expr, value: ast.expr):
        """Assign a variable in the current stack frame."""
        try:
            self.scope.assign(name, value)
        except TypeError as e:
            Log.error(f"Assignment type not supported at {self.error_loc(name)}: {e}")
            exit(1)
        except Exception as e:
            raise e

    def look_up(self, name: ast.expr) -> Optional[ast.expr]:
        """Look up a variable in the current stack frame."""
        return self.scope.look_up(name)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        with self:
            for arg in node.args.args:
                self.assign(ast.Name(id=arg.arg, ctx=ast.Load()), ast.Name(id=arg.arg, ctx=ast.Load()))
            for kwarg in node.args.kwonlyargs:
                self.assign(ast.Name(id=kwarg.arg, ctx=ast.Load()), ast.Name(id=kwarg.arg, ctx=ast.Load()))
            for body_node in node.body:
                self.visit(body_node)

    def visit_If(self, node: ast.If):
        with self.path(node.test):
            for body_node in node.body:
                self.visit(body_node)

        with self.path(ast.UnaryOp(op=ast.Not(), operand=node.test)):
            for orelse_node in node.orelse:
                self.visit(orelse_node)
    
    def visit_Assign(self, node: ast.Assign):
        """Handle variable assignments."""
        for target in node.targets:
            self.assign(target, node.value)
    
    def visit_Return(self, node: ast.Return):
        if not self.current_handler:
            raise ValueError("No current handler set for return statement.")
        if not self.graph:
            raise ValueError("No graph set for return statement.")
        if not node.value:
            raise ValueError("Return statement must have a value. ()")
        succ = self._extract_successor(node.value)
        if not succ:
            return
        assert self.current_node is not None, "No current procedure node set for return statement."
        self.graph.add_node(succ)
        edge = ProcedureEdge(
            from_procedure=self.current_node,
            to_procedure=succ,
            event_type=self.current_handler.event_type,
            condition=self.current_guard()
        )
        self.graph.add_edge(edge)
        self.next_neighbor.append(succ)


    def error_loc(self, node) -> str:
        """Get the location of an AST node for error reporting."""
        return f"line {node.lineno}, column {node.col_offset} in {inspect.getfile(self.ctx)}"

    def construct_procedure_graph(self, entry: Procedure[Sigma]) -> ProcedureGraph:
        # construct the node for entry procedure
        entry_node = self._create_procedure_node(entry.__class__, str(entry.prompt))
        worklist = [entry_node]
        self.graph.add_node(entry_node)
        self.graph.entry_point = entry_node
        visited = set()
        # Construct by BFS
        while worklist:
            now, *worklist = worklist
            if now.to_key() in visited:
                continue
            visited.add(now.to_key())
            self.current_node = now
            self.next_neighbor = []
            for handler_info in now.handlers.values():
                self.current_handler = handler_info
                # Get the handler method from the procedure class
                handler_method = getattr(now.procedure, handler_info.method_name, None)
                if not handler_method:
                    Log.warning(f"Handler method {handler_info.method_name} not found in {now.name}.")
                    continue
                
                # Analyze the handler method's AST
                handler_ast = ast.parse(textwrap.dedent(inspect.getsource(handler_method)))
                self.visit(handler_ast)
                
                # Add all successors to the worklist
                for succ in self.next_neighbor:
                    worklist.append(succ)
        return self.graph

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

    def _extract_successor(self, node: ast.AST) -> Optional[ProcedureNode]:
        """Extract procedure types from AST nodes."""
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
                proc_class = getattr(self.ctx, class_name, None)
                if not proc_class or not issubclass(proc_class, Procedure):
                    return None
                return self._create_procedure_node(proc_class, prompt_txt if prompt_txt else str(proc_class.prompt))
        
        elif isinstance(node, ast.Name):
            # return `self`
            if node.id == 'self':
                assert self.current_node is not None, "No current procedure node set for self reference."
                # Use the current procedure node's class and prompt text
                return self.current_node
            # Direct class reference
            proc_class = getattr(self.ctx, node.id, None)
            if proc_class and issubclass(proc_class, Procedure):
                # If it's a procedure class, add it with its default prompt
                return self._create_procedure_node(proc_class, str(proc_class.prompt))
        elif isinstance(node, ast.JoinedStr | ast.Constant):
            return None

        Log.warning(f"Cannot extract procedure from AST node: {ast.dump(node)} at {self.error_loc(node)}")
        return None


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
