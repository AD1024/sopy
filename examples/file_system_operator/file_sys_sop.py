import os
from sopy import Procedure, handler, End, make_prompt, Prompt, SOP
from .events import *

class FSInfo:
    def __init__(self, src_path: str, dst_path: str, backup_path: str):
        self.src_path = src_path
        self.dst_path = dst_path
        self.backup_path = backup_path
        self.path_read = set()
        self.path_copied = set()
        self.path_deleted = set()

"""
An SOP that provides the atomic file migration operations
ReadDir:
    Instruction: Read the directory to be migrated, then goto BackUp.
    Action: eReadDirectory

Backup:
    Instruction: For each file in the directory, copy it to the given backup location.
                 If copy fails, try up to 3 times; otherwise, abort the migration.
    Action: eFileCopied

Copy:
    Instruction: For each file in the directory, copy it to the migrate location.
    Action: eFileCopied

Clean:
    Instruction: Delete the backup files and the original directory and finish the procedure.
    Action: eFileDeleted

Abort:
    Instruction: Delete the backup files if any were created then finish the procedure.
    Action: eFileDeleted
"""

class ReadDir(Procedure[FSInfo]):
    prompt = make_prompt("Read the directory to be migrated")

    @handler
    def handle_eReadDirectory(self, sigma: FSInfo, event: eReadDirectory):
        sigma.path_read |= set(event.payload[1])
        return BackUpCopy(prompt=make_prompt("For each file in the directory, copy it to the given backup location."))

class Copy(Procedure[FSInfo]):

    def __init__(self, prompt: Prompt):
        self.prompt = prompt
        self.copied = set()
        self.srcs = set()
        self.kv = dict()
        self.retry = 3

    @handler
    def handle_eCopyRequest(self, sigma: FSInfo, event: eCopyRequest):
        if self.retry <= 0:
            return End(make_prompt("Backup failed after 3 retries."))
        src, dst = event.payload
        assert os.path.abspath(src).startswith(os.path.abspath(sigma.src_path)), f"Source {src} is not from the source path {sigma.src_path}."
        assert src in sigma.path_read, f"Source {src} not in read paths."
        assert dst not in self.copied, f"File {dst} already exists in this backup operation for {self.kv[dst]}."
        assert src not in self.srcs, f"File {src} already copied in this backup operation."
        return self

class BackUpCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        src, dst = event.payload
        self.copied.add(src)
        self.kv[dst] = src
        if len(sigma.path_read) == len(self.copied) and sigma.path_read == sigma.path_copied:
            return MigrateCopy(prompt=make_prompt("For each file in the directory, copy it to the migrate location."))

class MigrateCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        src, dst = event.payload
        self.copied.add(src)
        self.kv[dst] = src
        if len(sigma.path_read) == len(self.copied) and sigma.path_read == sigma.path_copied:
            return Clear()

class Clear(Procedure[FSInfo]):
    prompt = make_prompt("Delete the backup files and the original directory and finish the procedure.")

    def __init__(self):
        self.deleted = set()

    @handler
    def handle_eFileDeleteRequest(self, sigma: FSInfo, event: eFileDeleteRequest):
        dst = event.payload
        assert dst not in self.deleted, f"File {dst} already deleted in this backup operation."
        assert os.path.abspath(dst).startswith(os.path.abspath(sigma.backup_path)), f"File {dst} is not from the backup path {sigma.backup_path}."

    @handler
    def handle_eFileDeleted(self, sigma: FSInfo, event: eFileDeleted):
        dst = event.payload
        if dst is None:
            return f"Failed to delete file {dst}. Please try again."
        assert dst not in self.deleted, f"File {dst} already deleted in the backup path."
        sigma.path_deleted.add(dst)
        if len(sigma.path_deleted) == len(sigma.path_copied):
            return End(make_prompt("All files deleted successfully."))

class FileSystemOperator(SOP[FSInfo]):
    def __init__(self, src_path: str, dst_path: str, backup_path: str):
        """
        Initialize the SOP with the given source and destination paths.
        """
        self.state = FSInfo(src_path, dst_path, backup_path)
        self.current_proc = ReadDir()
        self.ignore_unrecongnized_events = False