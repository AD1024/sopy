#!/usr/bin/env python3
"""
Test script for the file system simulator.
Demonstrates the usage of touch, mkdir, copy, delete, and other file system operations.
"""

from file_sys_sim import FileSystem

def main():
    # Create a new file system
    fs = FileSystem()
    
    print("=== File System Simulator Test ===\n")
    
    # Test initial state
    print("1. Initial state:")
    print(f"Current directory: {fs.pwd()}")
    print(fs.ls())
    print()
    
    # Test mkdir
    print("2. Creating directories:")
    print(fs.mkdir("home"))
    print(fs.mkdir("/home/user"))
    print(fs.mkdir("home/user/documents"))
    print(fs.mkdir("home/user/downloads"))
    print(fs.ls("/"))
    print()
    
    # Test touch
    print("3. Creating files:")
    print(fs.touch("/home/user/readme.txt"))
    print(fs.touch("home/user/documents/notes.txt"))
    print(fs.touch("/home/user/documents/todo.txt"))
    print(fs.ls("/home/user"))
    print()
    
    # Test cd and pwd
    print("4. Changing directories:")
    print(fs.cd("/home/user/documents"))
    print(f"Current directory: {fs.pwd()}")
    print(fs.ls())
    print()
    
    # Test relative paths
    print("5. Using relative paths:")
    print(fs.touch("../config.ini"))
    print(fs.mkdir("../pictures"))
    print(fs.ls(".."))
    print()
    
    # Test copy file
    print("6. Copying files:")
    print(fs.copy("notes.txt", "notes_backup.txt"))
    print(fs.copy("/home/user/readme.txt", "/home/user/documents/readme_copy.txt"))
    print(fs.ls())
    print()
    
    # Test copy directory
    print("7. Copying directories:")
    print(fs.copy("/home/user/documents", "/home/user/docs_backup"))
    print(fs.ls("/home/user"))
    print()
    print("Contents of copied directory:")
    print(fs.ls("/home/user/docs_backup"))
    print()
    
    # Test delete file
    print("8. Deleting files:")
    print(fs.delete("notes_backup.txt"))
    print(fs.delete("/home/user/config.ini"))
    print(fs.ls())
    print()
    
    # Test delete directory
    print("9. Deleting directories:")
    print(fs.delete("/home/user/downloads"))
    print(fs.ls("/home/user"))
    print()
    
    # Test error cases
    print("10. Testing error cases:")
    print(fs.touch("/nonexistent/file.txt"))  # Parent doesn't exist
    print(fs.mkdir("notes.txt"))  # Already exists as file
    print(fs.copy("/nonexistent", "/somewhere"))  # Source doesn't exist
    print(fs.delete("/nonexistent"))  # Path doesn't exist
    print(fs.cd("notes.txt"))  # Not a directory
    print()
    
    # Final state
    print("11. Final state:")
    print(f"Current directory: {fs.pwd()}")
    print(fs.ls("/"))


if __name__ == "__main__":
    main()
