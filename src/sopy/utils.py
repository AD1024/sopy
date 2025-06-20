from enum import Enum
from typing import Dict, Any, List
from datetime import datetime
import sys


class LogLevel(Enum):
    """Enumeration for different log levels"""
    VERBOSE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class ColorCodes:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColorLogger:
    """A colorful logging class that prints logs with different colors for different levels"""
    
    def __init__(self, enable_colors: bool = True, min_level: LogLevel = LogLevel.INFO):
        """
        Initialize the ColorLogger
        
        Args:
            enable_colors: Whether to use colors in output (default: True)
            min_level: Minimum log level to display (default: INFO)
        """
        self.enable_colors = enable_colors
        self.min_level = min_level
        
        # Color mapping for each log level
        self.level_colors = {
            LogLevel.VERBOSE: ColorCodes.BRIGHT_BLACK,
            LogLevel.DEBUG: ColorCodes.CYAN,
            LogLevel.INFO: ColorCodes.GREEN,
            LogLevel.WARNING: ColorCodes.YELLOW,
            LogLevel.ERROR: ColorCodes.RED
        }
        
        # Level names
        self.level_names = {
            LogLevel.VERBOSE: "VERBOSE",
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR"
        }
    
    def _format_message(self, level: LogLevel, message: str, include_timestamp: bool = True) -> str:
        """
        Format a log message with color, timestamp, and proper multi-line indentation
        
        Args:
            level: The log level
            message: The message to log
            include_timestamp: Whether to include timestamp (default: True)
            
        Returns:
            Formatted message string with proper multi-line handling
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if include_timestamp else ""
        level_name = self.level_names[level]
        
        # Split message into lines for multi-line handling
        lines = message.split('\n')
        
        if self.enable_colors:
            color = self.level_colors[level]
            formatted_level = f"{color}{ColorCodes.BOLD}[{level_name}]{ColorCodes.RESET}"
            
            if include_timestamp:
                # Calculate indentation for subsequent lines
                indent_length = len(timestamp) + len(f" [{level_name}] ")
                indent = " " * indent_length
                
                # Format first line with timestamp and level
                formatted_lines = [f"{ColorCodes.BRIGHT_BLACK}{timestamp}{ColorCodes.RESET} {formatted_level} {lines[0]}"]
                
                # Format subsequent lines with proper indentation
                for line in lines[1:]:
                    if line.strip():  # Only add indentation to non-empty lines
                        formatted_lines.append(f"{ColorCodes.BRIGHT_BLACK}{' ' * len(timestamp)}{ColorCodes.RESET} {' ' * (len(f'[{level_name}]') + 1)} {line}")
                    else:
                        formatted_lines.append("")
                
                return '\n'.join(formatted_lines)
            else:
                # No timestamp, simpler indentation
                indent_length = len(f"[{level_name}] ")
                indent = " " * indent_length
                
                # Format first line with level
                formatted_lines = [f"{formatted_level} {lines[0]}"]
                
                # Format subsequent lines with proper indentation
                for line in lines[1:]:
                    if line.strip():  # Only add indentation to non-empty lines
                        formatted_lines.append(f"{' ' * indent_length}{line}")
                    else:
                        formatted_lines.append("")
                
                return '\n'.join(formatted_lines)
        else:
            # No colors
            if include_timestamp:
                # Calculate indentation for subsequent lines
                indent_length = len(timestamp) + len(f" [{level_name}] ")
                
                # Format first line with timestamp and level
                formatted_lines = [f"{timestamp} [{level_name}] {lines[0]}"]
                
                # Format subsequent lines with proper indentation
                for line in lines[1:]:
                    if line.strip():  # Only add indentation to non-empty lines
                        formatted_lines.append(f"{' ' * indent_length}{line}")
                    else:
                        formatted_lines.append("")
                
                return '\n'.join(formatted_lines)
            else:
                # No timestamp, simpler indentation
                indent_length = len(f"[{level_name}] ")
                
                # Format first line with level
                formatted_lines = [f"[{level_name}] {lines[0]}"]
                
                # Format subsequent lines with proper indentation
                for line in lines[1:]:
                    if line.strip():  # Only add indentation to non-empty lines
                        formatted_lines.append(f"{' ' * indent_length}{line}")
                    else:
                        formatted_lines.append("")
                
                return '\n'.join(formatted_lines)
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if a message should be logged based on minimum level"""
        return level.value >= self.min_level.value
    
    def _print_log(self, level: LogLevel, message: str, file=None, include_timestamp: bool = True):
        """Print a log message if it meets the minimum level requirement"""
        if self._should_log(level):
            formatted_msg = self._format_message(level, message, include_timestamp)
            print(formatted_msg, file=file or sys.stdout)
    
    def verbose(self, message: str, include_timestamp: bool = True):
        """Log a verbose message (lowest priority)"""
        self._print_log(LogLevel.VERBOSE, message, include_timestamp=include_timestamp)
    
    def debug(self, message: str, include_timestamp: bool = True):
        """Log a debug message"""
        self._print_log(LogLevel.DEBUG, message, include_timestamp=include_timestamp)
    
    def info(self, message: str, include_timestamp: bool = True):
        """Log an info message"""
        self._print_log(LogLevel.INFO, message, include_timestamp=include_timestamp)
    
    def warning(self, message: str, include_timestamp: bool = True):
        """Log a warning message"""
        self._print_log(LogLevel.WARNING, message, include_timestamp=include_timestamp)
    
    def warn(self, message: str, include_timestamp: bool = True):
        """Alias for warning()"""
        self.warning(message, include_timestamp)
    
    def error(self, message: str, include_timestamp: bool = True):
        """Log an error message"""
        self._print_log(LogLevel.ERROR, message, file=sys.stderr, include_timestamp=include_timestamp)
    
    def set_min_level(self, level: LogLevel):
        """Set the minimum log level to display"""
        self.min_level = level
    
    def enable_color(self, enable: bool = True):
        """Enable or disable color output"""
        self.enable_colors = enable
    
    def log(self, level: LogLevel, message: str, include_timestamp: bool = True):
        """Generic log method that accepts any LogLevel"""
        self._print_log(level, message, include_timestamp=include_timestamp)


class Log:
    # Create a default logger instance for convenience
    logger = ColorLogger()

    @staticmethod
    def set_min_level(level: LogLevel):
        """Set the minimum log level for the default logger"""
        Log.logger.set_min_level(level)

    # Convenience functions for quick logging
    @staticmethod
    def verbose(message: str, include_timestamp: bool = False):
        """Quick verbose logging function"""
        Log.logger.verbose(message, include_timestamp)

    @staticmethod
    def debug(message: str, include_timestamp: bool = False):
        """Quick debug logging function"""
        Log.logger.debug(message, include_timestamp)

    @staticmethod
    def info(message: str, include_timestamp: bool = False):
        """Quick info logging function"""
        Log.logger.info(message, include_timestamp)

    @staticmethod
    def warning(message: str, include_timestamp: bool = False):
        """Quick warning logging function"""
        Log.logger.warning(message, include_timestamp)

    @staticmethod
    def error(message: str, include_timestamp: bool = False):
        """Quick error logging function"""
        Log.logger.error(message, include_timestamp)
