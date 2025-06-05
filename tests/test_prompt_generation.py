#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '..')

from sopy.prompt.prompt_generator import make_prompt_from_procedure
from examples.file_system_operator.file_sys_sop import ReadDir

def test_prompt_generation():
    """Test the SOP prompt generation from the file system operator example."""
    
    # Create an entry procedure
    entry_procedure = ReadDir()
    
    # Generate the SOP prompt
    objective = "Migrate files from source to destination with backup safety"
    prompt = make_prompt_from_procedure(entry_procedure, objective)
    
    print("Generated SOP Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

if __name__ == "__main__":
    test_prompt_generation()
