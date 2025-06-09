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
    ctx: Dict[int, ast.expr]
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
        def __init__(self, ctx: Dict[int, ast.expr] = {}, parent: Optional[Self] = None,
                     path_condition: Optional[ast.expr] = None, escape: bool = False):
            self.parent: Optional[Self] = parent
            # map from expr hash to a tuple of (value, condition of assignment)
            self.variables: Dict[int, ast.expr] = ctx
            self.path_condition = path_condition
            self.escape = escape

        @staticmethod
        @lru_cache
        def name_hash(name: ast.expr) -> int:
            This = HandlerAnalysis.AnalysisScope
            if isinstance(name, ast.Name):
                return hash(name.id)
            elif isinstance(name, ast.Attribute):
                return hash((name.attr, This.name_hash(name.value)))
            elif isinstance(name, ast.Subscript):
                return hash((This.name_hash(name.value), This.name_hash(name.slice)))
            elif isinstance(name, ast.List):
                return hash(tuple(This.name_hash(e) for e in name.elts))
            elif isinstance(name, ast.Tuple):
                return hash(tuple(This.name_hash(e) for e in name.elts))
            elif isinstance(name, ast.Constant):
                return hash(name.value)
            raise TypeError(f"Unsupported name type for hashing during analysis: {type(name)}")
    
        def set_var(self, name: ast.expr, value: ast.expr):
            name_hash = self.name_hash(name)
            self.variables[name_hash] = value
        
        def get_var(self, name: ast.expr) -> Optional[ast.expr]:
            """Get a variable value by its name."""
            name_hash = self.name_hash(name)
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
            name_hash = self.name_hash(name)
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
    
    def set_context(self, ctx: Dict[int, ast.expr]):
        assert self.scope.parent is None, "Cannot set context in a non-root scope."
        self.scope = HandlerAnalysis.AnalysisScope(ctx=ctx, escape=False)

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
                self.set_context(handler_info.ctx)
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
                    ctx={},
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


def make_prompt_from_procedure(entry_procedure: Procedure[Sigma], objective: str = "") -> str:
    """
    Generate a natural language system prompt from a ProcedureGraph.
    
    Args:
        entry_procedure: The starting procedure for the SOP
        objective: Optional description of the overall objective
    
    Returns:
        A formatted SOP prompt string
    """
    # Build the procedure graph
    builder = ProcedureGraphBuilder()
    graph = builder.build_graph(entry_procedure)
    
    # Generate the prompt
    generator = SOPPromptGenerator()
    return generator.generate_prompt(graph, objective)


class SOPPromptGenerator:
    """Generates natural language SOP prompts from procedure graphs."""
    
    def __init__(self):
        self.visited_nodes = set()
        self.step_counter = 1
        
    def generate_prompt(self, graph: ProcedureGraph, objective: str = "") -> str:
        """Generate the complete SOP prompt."""
        if not graph.entry_point:
            return "No entry point found in procedure graph."
        
        # Generate header
        header = "Here is the standard operating procedure (SOP)"
        if objective:
            header += f" for {objective}"
        header += ":\n"
        
        # Generate steps using topological order
        self.visited_nodes = set()
        self.step_counter = 1
        steps = self._generate_steps_topological(graph)
        
        return header + "\n".join(steps)
    
    def _get_topological_order(self, graph: ProcedureGraph) -> List[ProcedureNode]:
        """Get nodes in topological order to avoid terminal nodes appearing before their predecessors."""
        # Use Kahn's algorithm for topological sorting
        in_degree = {}
        all_nodes = list(graph.nodes.values())
        
        # Initialize in-degree count
        for node in all_nodes:
            in_degree[node.to_key()] = 0
        
        # Calculate in-degrees
        for node_key, edges in graph.edges.items():
            for edge in edges:
                if edge.to_procedure.to_key() != edge.from_procedure.to_key():  # Skip self-loops
                    in_degree[edge.to_procedure.to_key()] = in_degree.get(edge.to_procedure.to_key(), 0) + 1
        
        # Start with nodes that have no incoming edges
        queue = [node for node in all_nodes if in_degree[node.to_key()] == 0]
        result = []
        
        while queue:
            # Get node with zero in-degree
            current = queue.pop(0)
            result.append(current)
            
            # Process all outgoing edges
            outgoing_edges = graph.edges.get((current.name, current.prompt_text), [])
            for edge in outgoing_edges:
                if edge.to_procedure.to_key() != current.to_key():  # Skip self-loops
                    in_degree[edge.to_procedure.to_key()] -= 1
                    if in_degree[edge.to_procedure.to_key()] == 0:
                        queue.append(edge.to_procedure)
        
        return result
    
    def _generate_steps_topological(self, graph: ProcedureGraph) -> List[str]:
        """Generate steps using topological ordering to ensure correct sequence."""
        steps = []
        
        # Get nodes in topological order
        ordered_nodes = self._get_topological_order(graph)
        
        # Process each node in topological order
        for node in ordered_nodes:
            if node.to_key() in self.visited_nodes:
                continue
                
            self.visited_nodes.add(node.to_key())
            
            # Skip empty prompt nodes
            if not node.prompt_text.strip():
                continue
            
            # Get outgoing edges
            outgoing_edges = graph.edges.get((node.name, node.prompt_text), [])
            
            # Detect pattern
            pattern = self._detect_transition_pattern(node, outgoing_edges)
            
            # Skip self-loop only patterns
            if pattern == "self_loop_only":
                continue
            
            # Generate step text
            step_text = self._generate_step_text(node, outgoing_edges, pattern)
            if step_text.strip():  # Only add non-empty steps
                steps.append(step_text)
        
        return steps
    
    def _generate_steps(self, graph: ProcedureGraph, node: ProcedureNode) -> List[str]:
        """Generate steps starting from the given node."""
        steps = []
        
        if node.to_key() in self.visited_nodes:
            return steps
        
        self.visited_nodes.add(node.to_key())
        
        # Skip empty prompt nodes
        if not node.prompt_text.strip():
            # Process successors without creating a step
            for edge in graph.edges.get((node.name, node.prompt_text), []):
                if edge.to_procedure != node:
                    successor_steps = self._generate_steps(graph, edge.to_procedure)
                    steps.extend(successor_steps)
            return steps
        
        # Get outgoing edges
        outgoing_edges = graph.edges.get((node.name, node.prompt_text), [])
        
        # Detect pattern
        pattern = self._detect_transition_pattern(node, outgoing_edges)
        
        # Skip self-loop only patterns
        if pattern == "self_loop_only":
            return steps
        
        # Generate step text
        step_text = self._generate_step_text(node, outgoing_edges, pattern)
        if step_text.strip():  # Only add non-empty steps
            steps.append(step_text)
        
        # Process successor nodes with better deduplication
        processed_successors = set()
        for edge in outgoing_edges:
            if (edge.to_procedure != node and 
                edge.to_procedure.to_key() not in processed_successors and
                edge.to_procedure.to_key() not in self.visited_nodes):
                processed_successors.add(edge.to_procedure.to_key())
                successor_steps = self._generate_steps(graph, edge.to_procedure)
                steps.extend(successor_steps)
        
        return steps
    
    def _detect_transition_pattern(self, node: ProcedureNode, outgoing_edges: List[ProcedureEdge]) -> str:
        """Detect the pattern type based on current node and its outgoing transitions."""
        
        # Categorize edges
        self_loops = [e for e in outgoing_edges if e.to_procedure == node]
        non_self_edges = [e for e in outgoing_edges if e.to_procedure != node]
        terminal_edges = [e for e in non_self_edges if e.to_procedure.is_terminal]
        
        # Pattern detection
        if len(non_self_edges) == 0 and len(self_loops) > 0:
            return "self_loop_only"
        elif len(non_self_edges) == 1 and len(self_loops) == 0:
            return "sequential_flow"
        elif len(non_self_edges) > 1:
            return "conditional_branching"
        elif len(self_loops) > 0 and len(non_self_edges) > 0:
            return "retry_with_progression"
        elif len(terminal_edges) > 0:
            return "terminal_transition"
        else:
            return "complex_pattern"
    
    def _generate_step_text(self, node: ProcedureNode, outgoing_edges: List[ProcedureEdge], pattern: str) -> str:
        """Generate the step text for a given node and pattern."""
        if node.name == "End":
            return ""
        step_text = f"Procedure {node.name}:\n"
        
        # Ensure proper punctuation for the instruction - remove trailing periods first
        instruction = node.prompt_text.strip().rstrip('.')
        step_text += f"    Instruction: {instruction}"
        
        # Add pattern-specific transition description
        transition_text = ""
        if pattern == "sequential_flow":
            transition_text = self._translate_sequential_flow(outgoing_edges)
        elif pattern == "conditional_branching":
            transition_text = self._translate_conditional_branching(outgoing_edges)
        elif pattern == "retry_with_progression":
            transition_text = self._translate_retry_pattern(node, outgoing_edges)
        elif pattern == "terminal_transition":
            transition_text = self._translate_terminal_transitions(outgoing_edges)
        
        # Add transition text and ensure proper punctuation
        if transition_text:
            step_text += transition_text
        
        # Ensure the instruction ends with a period
        if not step_text.rstrip().endswith('.'):
            step_text += '.'
        
        # Add action information
        actions = [edge.event_type.__name__ for edge in outgoing_edges if edge.event_type]
        unique_actions = list(set(actions))
        if unique_actions:
            step_text += f"\n    Action: {', '.join(unique_actions)}"
        
        self.step_counter += 1
        return step_text
    
    def _translate_sequential_flow(self, edges: List[ProcedureEdge]) -> str:
        """Translate sequential flow pattern."""
        if not edges:
            return ""
        
        edge = edges[0]
        target_text = self._get_meaningful_target_description(edge)
        
        if not target_text:
            return ""
            
        if edge.condition:
            condition_text = self._translate_conditions(edge.condition)
            if condition_text and condition_text != "the condition is met":
                return f" If {condition_text}, then {target_text}."
        else:
            # For unconditional sequential flow, just mention it proceeds to next step
            return f" Then proceed to {edge.to_procedure.name}."
        Log.warning(f"Unexpected edge condition: {edge.condition} for edge {edge.from_procedure.name} -> {edge.to_procedure.name}")
        return ""
    
    def _translate_conditional_branching(self, edges: List[ProcedureEdge]) -> str:
        """Translate conditional branching pattern."""
        if not edges:
            return ""
        
        # Filter out self-loops and group by target
        non_self_edges = [e for e in edges if e.to_procedure.prompt_text != e.from_procedure.prompt_text]
        if not non_self_edges:
            return ""
        
        conditions = []
        else_edges = []
        
        for edge in non_self_edges:
            # Get target description first
            target_text = self._get_meaningful_target_description(edge)
            
            # Only process edges with meaningful targets
            if target_text:
                if edge.condition:
                    condition_text = self._translate_conditions(edge.condition)
                    # if condition_text and condition_text != "the condition is met":
                    #     conditions.append(f"If {condition_text}, then {target_text}")
                    if condition_text:
                        conditions.append(f"If {condition_text}, then go to {edge.to_procedure.name}")
                else:
                    else_edges.append(edge)
        
        # Add else cases (only if we have meaningful targets)
        for edge in else_edges:
            target_text = self._get_meaningful_target_description(edge)
            if target_text:
                conditions.append(f"Otherwise, go to {edge.to_procedure.name}")
        
        if conditions:
            return ". " + ". ".join(conditions) + "."
        return ""
    
    def _get_meaningful_target_description(self, edge: ProcedureEdge) -> Optional[str]:
        """Get a meaningful description of the target procedure, avoiding repetition."""
        target_prompt = edge.to_procedure.prompt_text.strip()
        current_prompt = edge.from_procedure.prompt_text.strip()
        
        # For terminal nodes, always use special wording regardless of prompt
        if edge.to_procedure.is_terminal:
            return "finish the procedure"
            
        # Skip if target prompt is empty or the same as current
        if not target_prompt or target_prompt == current_prompt:
            return None
            
        # Use the target prompt, but make it concise
        return target_prompt.lower()
    
    def _translate_retry_pattern(self, node: ProcedureNode, edges: List[ProcedureEdge]) -> str:
        """Translate retry with progression pattern."""
        self_edges = [e for e in edges if e.to_procedure == node]
        progression_edges = [e for e in edges if e.to_procedure != node]
        
        parts = []
        
        # Add retry conditions
        retry_conditions = []
        for edge in self_edges:
            if edge.condition:
                condition_text = self._translate_conditions(edge.condition)
                retry_conditions.append(condition_text)
        
        if retry_conditions:
            parts.append(f"If {' or '.join(retry_conditions)}, try again")
        
        # Add progression conditions
        for edge in progression_edges:
            if edge.condition:
                condition_text = self._translate_conditions(edge.condition)
                action_text = edge.to_procedure.prompt_text.lower()
                parts.append(f"If {condition_text}, then {action_text}")
        
        if parts:
            return ". " + ". ".join(parts) + "."
        return ""
    
    def _translate_terminal_transitions(self, edges: List[ProcedureEdge]) -> str:
        """Translate transitions to terminal nodes."""
        terminal_edges = [e for e in edges if e.to_procedure.is_terminal]
        
        if not terminal_edges:
            return ""
        
        conditions = []
        for edge in terminal_edges:
            if edge.condition:
                condition_text = self._translate_conditions(edge.condition)
                conditions.append(f"If {condition_text}, then finish the procedure")
            else:
                conditions.append("Then finish the procedure")
        
        if conditions:
            return ". " + ". ".join(conditions)
        return ""
    
    def _translate_conditions(self, conditions: List[ast.expr]) -> str:
        """Translate a list of AST conditions to natural language."""
        if not conditions:
            return ""
        
        translated = []
        for condition in conditions:
            translated.append(self._translate_single_condition(condition))
        
        return " and ".join(translated)
    
    def _translate_single_condition(self, condition: ast.expr) -> str:
        """Translate a single AST condition to natural language."""
        try:
            if isinstance(condition, ast.Call):
                # Handle cond() function calls
                if (isinstance(condition.func, ast.Name) and 
                    condition.func.id == 'cond' and 
                    len(condition.args) >= 2 and 
                    isinstance(condition.args[1], ast.Constant)):
                    return condition.args[1].value
                
            elif isinstance(condition, ast.Compare):
                return self._translate_comparison(condition)
            
            elif isinstance(condition, ast.BoolOp):
                return self._translate_bool_op(condition)
            
            elif isinstance(condition, ast.UnaryOp):
                return self._translate_unary_op(condition)
            
            elif isinstance(condition, ast.Name):
                return self._translate_name(condition)
            
            elif isinstance(condition, ast.Attribute):
                return self._translate_attribute(condition)
            
            # Fallback: convert to source code
            return ast.unparse(condition)
            
        except Exception as e:
            Log.warning(f"Failed to translate condition: {e}")
            return "the condition is met"
    
    def _translate_comparison(self, node: ast.Compare) -> str:
        """Translate comparison operations."""
        left = self._translate_expr_to_text(node.left)
        
        parts = [left]
        for op, right in zip(node.ops, node.comparators):
            op_text = self._translate_operator(op)
            right_text = self._translate_expr_to_text(right)
            parts.append(f"{op_text} {right_text}")
        
        return " ".join(parts)
    
    def _translate_bool_op(self, node: ast.BoolOp) -> str:
        """Translate boolean operations (and, or)."""
        op_text = "and" if isinstance(node.op, ast.And) else "or"
        conditions = [self._translate_single_condition(value) for value in node.values]
        return f" {op_text} ".join(conditions)
    
    def _translate_unary_op(self, node: ast.UnaryOp) -> str:
        """Translate unary operations (not)."""
        if isinstance(node.op, ast.Not):
            operand = self._translate_single_condition(node.operand)
            return f"not {operand}"
        return ast.unparse(node)
    
    def _translate_name(self, node: ast.Name) -> str:
        """Translate variable names."""
        return node.id
    
    def _translate_attribute(self, node: ast.Attribute) -> str:
        """Translate attribute access with potential Pydantic Field descriptions."""
        # Try to get field description if available
        field_desc = self._get_field_description(node)
        if field_desc:
            return field_desc
        
        # Fallback to readable attribute name
        value_text = self._translate_expr_to_text(node.value)
        return f"{value_text}.{node.attr}"
    
    def _translate_operator(self, op: ast.cmpop) -> str:
        """Translate comparison operators."""
        op_map = {
            ast.Eq: "is",
            ast.NotEq: "is not",
            ast.Lt: "is less than",
            ast.LtE: "is less than or equal to", 
            ast.Gt: "is greater than",
            ast.GtE: "is greater than or equal to",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "is in",
            ast.NotIn: "is not in"
        }
        return op_map.get(type(op), "==")
    
    def _translate_expr_to_text(self, expr: ast.expr) -> str:
        """Translate an expression to readable text."""
        if isinstance(expr, ast.Name):
            return expr.id
        elif isinstance(expr, ast.Attribute):
            field_desc = self._get_field_description(expr)
            if field_desc:
                return field_desc
            value_text = self._translate_expr_to_text(expr.value)
            return f"{value_text}.{expr.attr}"
        elif isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return f"'{expr.value}'"
            return str(expr.value)
        else:
            return ast.unparse(expr)
    
    def _get_field_description(self, attr_node: ast.Attribute) -> Optional[str]:
        """Try to extract Pydantic Field description from attribute access."""
        try:
            # Try to resolve the field description from the actual class definition
            return self._resolve_field_description(attr_node)
        except Exception as e:
            Log.warning(f"Failed to resolve field description for {ast.unparse(attr_node)}: {e}")
            return None
    
    def _resolve_field_description(self, attr_node: ast.Attribute) -> Optional[str]:
        """Resolve field description by inspecting the class definition."""
        # This is a more sophisticated approach that tries to resolve the actual Field definitions
        field_name = attr_node.attr
        
        # Try to determine the type of the object being accessed
        # For now, we'll look for common patterns like sigma.field_name
        if isinstance(attr_node.value, ast.Name):
            var_name = attr_node.value.id
            
            # Look for type annotations in function signatures that might give us clues
            # This is a simplified heuristic - in practice you'd need full type resolution
            if var_name in ['sigma', 'state']:
                # Try to find Sigma type or similar in the current context
                field_desc = self._find_field_in_common_types(field_name)
                if field_desc:
                    return field_desc
        
        return None
    
    def _find_field_in_common_types(self, field_name: str) -> Optional[str]:
        """Find field description in commonly used types."""
        # This is a heuristic approach - we look for classes with Field definitions
        # In practice, you'd want full type resolution
        import sys
        
        # Look through loaded modules for BaseModel classes
        for _, module in sys.modules.items():
            if not module or not hasattr(module, '__dict__'):
                continue
                
            try:
                for attr_name in dir(module):
                    attr_value = getattr(module, attr_name, None)
                    if (isinstance(attr_value, type) and 
                        hasattr(attr_value, '__annotations__') and
                        hasattr(attr_value, '__dict__')):
                        
                        # Check if this looks like a BaseModel class
                        if self._is_likely_basemodel(attr_value):
                            desc = self._extract_field_description_from_class(attr_value, field_name)
                            if desc:
                                return desc
            except Exception:
                continue
                
        return None
    
    def _is_likely_basemodel(self, cls: type) -> bool:
        """Check if a class is likely a BaseModel with Field definitions."""
        try:
            # Check if it has the typical BaseModel characteristics
            if hasattr(cls, '__fields__'):  # Pydantic v1
                return True
            if hasattr(cls, 'model_fields'):  # Pydantic v2
                return True
            # Check for common BaseModel method names
            if hasattr(cls, 'model_validate') or hasattr(cls, 'parse_obj'):
                return True
            # Check class hierarchy
            for base in getattr(cls, '__mro__', []):
                if base.__name__ == 'BaseModel':
                    return True
        except Exception:
            pass
        return False
    
    def _extract_field_description_from_class(self, cls: type, field_name: str) -> Optional[str]:
        """Extract field description from a BaseModel class."""
        try:
            # Pydantic v2 approach
            if hasattr(cls, 'model_fields') and field_name in cls.model_fields:
                field_info = cls.model_fields[field_name]
                if hasattr(field_info, 'description') and field_info.description:
                    return field_info.description
            
            # Pydantic v1 approach
            if hasattr(cls, '__fields__') and field_name in cls.__fields__:
                field_info = cls.__fields__[field_name]
                if hasattr(field_info, 'field_info') and hasattr(field_info.field_info, 'description'):
                    return field_info.field_info.description
            
            # Fallback: inspect the class source for Field definitions
            return self._extract_field_from_source(cls, field_name)
            
        except Exception as e:
            Log.warning(f"Failed to extract field description from {cls.__name__}.{field_name}: {e}")
            return None
    
    def _extract_field_from_source(self, cls: type, field_name: str) -> Optional[str]:
        """Extract field description by parsing the class source code."""
        try:
            import inspect
            source = inspect.getsource(cls)
            # Parse the source to find Field definitions
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if node.target.id == field_name and node.value:
                        # Check if the value is a Field() call
                        if isinstance(node.value, ast.Call):
                            desc = self._extract_description_from_field_call(node.value)
                            if desc:
                                return desc
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == field_name:
                            if isinstance(node.value, ast.Call):
                                desc = self._extract_description_from_field_call(node.value)
                                if desc:
                                    return desc
        except Exception as e:
            Log.warning(f"Failed to parse source for {cls.__name__}.{field_name}: {e}")
            
        return None
    
    def _extract_description_from_field_call(self, call_node: ast.Call) -> Optional[str]:
        """Extract description from a Field() call in AST."""
        try:
            # Check if this is a Field call
            if isinstance(call_node.func, ast.Name) and call_node.func.id == 'Field':
                # Look for description in keyword arguments
                for keyword in call_node.keywords:
                    if keyword.arg == 'description' and isinstance(keyword.value, ast.Constant):
                        return keyword.value.value
                        
            return None
        except Exception:
            return None
