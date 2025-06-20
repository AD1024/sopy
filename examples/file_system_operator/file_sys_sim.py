from sopy.interface.strands import tool
from sopy import handler
from events import *


class INode:
    def __init__(self, name: str, parent: 'INode | None' = None, is_file: bool = False):
        self.name = name
        self.parent = parent
        self.children: list[INode] = []
        self.is_file = is_file

class FileSystem:
    def __init__(self):
        self.root = INode(name="/", is_file=False)
        self.current_dir = self.root
    
    def _find_node(self, path: str) -> INode | None:
        """Navigate to a node by absolute or relative path."""
        if path == "/":
            return self.root
        
        # Handle absolute vs relative paths
        if path.startswith("/"):
            current = self.root
            path = path[1:]  # Remove leading slash
        else:
            current = self.current_dir
        
        if not path:  # Empty path after removing slash
            return current
        
        # Split path and navigate
        parts = [p for p in path.split("/") if p]  # Filter out empty parts
        
        for part in parts:
            if part == "..":
                if current.parent:
                    current = current.parent
            elif part == ".":
                continue  # Stay in current directory
            else:
                # Find child with matching name
                found = False
                for child in current.children:
                    if child.name == part:
                        current = child
                        found = True
                        break
                if not found:
                    return None
        
        return current
    
    def _get_parent_and_name(self, path: str) -> tuple[INode | None, str]:
        """Split path into parent directory and filename/dirname."""
        if path == "/":
            return None, "/"
        
        # Split path into parent and name
        if "/" in path:
            parent_path, name = path.rsplit("/", 1)
            if not parent_path:  # Root directory case
                parent_path = "/"
        else:
            parent_path = "."
            name = path
        
        parent = self._find_node(parent_path)
        return parent, name
    
    def _copy_node_recursive(self, source_node: INode, target_parent: INode, new_name: str) -> INode:
        """Create a deep copy of a node and all its children."""
        # Create new node
        new_node = INode(name=new_name, parent=target_parent, is_file=source_node.is_file)
        target_parent.children.append(new_node)
        
        # If it's a directory, recursively copy children
        if not source_node.is_file:
            for child in source_node.children:
                self._copy_node_recursive(child, new_node, child.name)
        
        return new_node
    
    def touch(self, path: str) -> str:
        """Create a new file at the specified path."""
        parent, name = self._get_parent_and_name(path)
        
        if parent is None:
            return f"Error: Parent directory not found for path '{path}'"
        
        if parent.is_file:
            return f"Error: '{parent.name}' is not a directory"
        
        # Check if file already exists
        for child in parent.children:
            if child.name == name:
                if child.is_file:
                    return f"File '{path}' already exists"
                else:
                    return f"Error: '{path}' is a directory, not a file"
        
        # Create new file
        new_file = INode(name=name, parent=parent, is_file=True)
        parent.children.append(new_file)
        return f"Created file '{path}'"
    
    def mkdir(self, path: str) -> str:
        """Create a new directory at the specified path."""
        parent, name = self._get_parent_and_name(path)
        
        if parent is None:
            return f"Error: Parent directory not found for path '{path}'"
        
        if parent.is_file:
            return f"Error: '{parent.name}' is not a directory"
        
        # Check if directory already exists
        for child in parent.children:
            if child.name == name:
                if child.is_file:
                    return f"Error: '{path}' is a file, not a directory"
                else:
                    return f"Directory '{path}' already exists"
        
        # Create new directory
        new_dir = INode(name=name, parent=parent, is_file=False)
        parent.children.append(new_dir)
        return f"Created directory '{path}'"
    
    def delete(self, path: str) -> str:
        """Delete a file or directory at the specified path."""
        if path == "/":
            return "Error: Cannot delete root directory"
        
        node = self._find_node(path)
        if node is None:
            return f"Error: Path '{path}' not found"
        
        if node == self.current_dir:
            return f"Error: Cannot delete current directory '{path}'"
        
        # Remove from parent's children list
        if node.parent:
            node.parent.children.remove(node)
            if node.is_file:
                return f"Deleted file '{path}'"
            else:
                return f"Deleted directory '{path}'"
        
        return f"Error: Cannot delete '{path}'"
    
    def copy(self, source_path: str, dest_path: str) -> str:
        """Copy a file or directory from source to destination."""
        # Find source node
        source_node = self._find_node(source_path)
        if source_node is None:
            return f"Error: Source path '{source_path}' not found"
        
        if source_node == self.root:
            return "Error: Cannot copy root directory"
        
        # Determine destination parent and name
        dest_parent, dest_name = self._get_parent_and_name(dest_path)
        
        if dest_parent is None:
            return f"Error: Destination parent directory not found for path '{dest_path}'"
        
        if dest_parent.is_file:
            return f"Error: Destination parent '{dest_parent.name}' is not a directory"
        
        # Check if destination already exists
        for child in dest_parent.children:
            if child.name == dest_name:
                return f"Error: '{dest_path}' already exists"
        
        # Check for circular copy (copying directory into itself or subdirectory)
        if not source_node.is_file:
            current = dest_parent
            while current:
                if current == source_node:
                    return f"Error: Cannot copy directory '{source_path}' into itself or its subdirectory"
                current = current.parent
        
        # Perform the copy
        self._copy_node_recursive(source_node, dest_parent, dest_name)
        
        if source_node.is_file:
            return f"Copied file '{source_path}' to '{dest_path}'"
        else:
            return f"Copied directory '{source_path}' to '{dest_path}'"
    
    def ls(self, path: str | None = None) -> str:
        """List contents of a directory."""
        if path is None:
            node = self.current_dir
            path = self.pwd()
        else:
            node = self._find_node(path)
            if node is None:
                return f"Error: Path '{path}' not found"
        
        if node.is_file:
            return f"Error: '{path}' is a file, not a directory"
        
        if not node.children:
            return f"Directory '{path}' is empty"
        
        items = []
        for child in sorted(node.children, key=lambda x: x.name):
            if child.is_file:
                items.append(f"  {child.name} (file)")
            else:
                items.append(f"  {child.name}/ (directory)")
        
        return f"Contents of '{path}':\n" + "\n".join(items)
    
    def pwd(self) -> str:
        """Get current working directory path."""
        if self.current_dir == self.root:
            return "/"
        
        path_parts = []
        current = self.current_dir
        while current and current != self.root:
            path_parts.append(current.name)
            current = current.parent
        
        return "/" + "/".join(reversed(path_parts))
    
    def cd(self, path: str) -> str:
        """Change current directory."""
        node = self._find_node(path)
        if node is None:
            return f"Error: Directory '{path}' not found"
        
        if node.is_file:
            return f"Error: '{path}' is a file, not a directory"
        
        self.current_dir = node
        return f"Changed directory to '{self.pwd()}'"

_filesys = FileSystem()

# some initial setups
name_template = 'file_{}.txt'
_filesys.mkdir('/files')
_filesys.mkdir('/backups')
_filesys.mkdir('/migrated_files')

for i in range(3):
    _filesys.touch(f'/files/{name_template.format(i)}')

def is_file(path: str) -> bool:
    """Check if the given path is a file."""
    node = _filesys._find_node(path)
    return node is not None and node.is_file

@tool(eReadDirectory)
def read_dir(path: str):
    """Read the contents of `path` and return a list of file names. If `path` is not a directory, return None."""
    """This tool will trigger an eReadDirectory event."""
    node = _filesys._find_node(path)
    if node is None or node.is_file:
        return path, None
    return path, [child.name for child in node.children]

@tool(eFileCopied, eCopyRequest)
def cp(source: str, dest: str):
    """Copy a file from `source` to `dest`. If `source` does not exist, return None."""
    """This tool will trigger an eFileCopied event."""
    result = _filesys.copy(source, dest)
    if "Error" in result:
        return None
    return source, dest

@tool(eFileDeleted, eFileDeleteRequest)
def delete(path: str):
    """Delete a file or directory at `path`. If `path` does not exist, return None."""
    """This tool will trigger an eFileDeleted event."""
    result = _filesys.delete(path)
    if "Error" in result:
        return None
    return path
