import os
from pydantic import Field, BaseModel
from sopy import Procedure, handler, End, make_prompt, Prompt, SOP
from sopy.prompt import cond
from .events import *

class FSInfo(BaseModel):
    src_path: str = Field(..., description="The source path from which files are to be migrated.")
    dst_path: str = Field(..., description="The destination path to which files are to be migrated.")
    backup_path: str = Field(..., description="The backup path where files are temporarily stored during migration.")
    path_read: set[str] = Field(default_factory=set, description="Set of paths that have been read from the source directory.")
    path_copied: set[str] = Field(default_factory=set, description="Set of paths that have been copied to the destination directory.")
    path_deleted: set[str] = Field(default_factory=set, description="Set of paths that have been deleted from the backup directory.")
    path_backuped: set[str] = Field(default_factory=set, description="Set of paths that have been backed up to the backup directory.")

    def __init__(self, src_path: str, dst_path: str, backup_path: str):
        self.src_path = src_path
        self.dst_path = dst_path
        self.backup_path = backup_path

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
        if cond(self.retry == 0, "Retry limit reached"):
            return Abort()
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
        sigma.path_backuped.add(dst)
        if cond(sigma.path_read == sigma.path_copied,
                "All files copied to backup location."):
            return MigrateCopy(prompt=make_prompt("For each file in the directory, copy it to the migrate location."))
        return self

class MigrateCopy(Copy):
    @handler
    def handle_eFileCopied(self, sigma: FSInfo, event: eFileCopied):
        if event.payload is None:
            self.retry -= 1
            return self
        src, dst = event.payload
        self.copied.add(src)
        self.kv[dst] = src
        if cond(sigma.path_read == sigma.path_copied, "All files copied to destination location."):
            return Clear()
        return self

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
        if cond(sigma.path_deleted == sigma.path_copied, "All files deleted from backup location."):
            return End()
        return self
        
class Abort(Procedure[FSInfo]):
    prompt = make_prompt("Delete all copied backup files and migrated files (if any), and abort the procedure")

    def __init__(self):
        self.deleted_backup = set()
        self.deleted_migration = set()

    @handler
    def handle_eFileDeleteRequest(self, sigma: FSInfo, event: eFileDeleteRequest):
        # check whether the file is from the backup path or migration path
        dst = event.payload
        if os.path.abspath(dst).startswith(os.path.abspath(sigma.backup_path)):
            assert dst not in self.deleted_backup, f"File {dst} already deleted in the backup path."
        elif os.path.abspath(dst).startswith(os.path.abspath(sigma.dst_path)):
            assert dst not in self.deleted_migration, f"File {dst} already deleted in the migration path."
        else:
            return f"File {dst} is not from the backup or migration path. Cannot delete."
        return self

    @handler
    def handle_eFileDeleted(self, sigma: FSInfo, event: eFileDeleted):
        dst = event.payload
        if dst is None:
            return f"Failed to delete file {dst}. Please try again."
        if os.path.abspath(dst).startswith(os.path.abspath(sigma.backup_path)):
            self.deleted_backup.add(dst)
        elif os.path.abspath(dst).startswith(os.path.abspath(sigma.dst_path)):
            self.deleted_migration.add(dst)
        else:
            return f"File {dst} is not from the backup or migration path. Cannot delete."
        
        if cond(self.deleted_backup == sigma.path_backuped and self.deleted_migration == sigma.path_copied,
                "Backups and partially migrated files are deleted."):
            return End()
        return self


class FileSystemOperator(SOP[FSInfo]):
    def __init__(self, src_path: str, dst_path: str, backup_path: str):
        """
        Initialize the SOP with the given source and destination paths.
        """
        self.state = FSInfo(src_path, dst_path, backup_path)
        self.current_proc = ReadDir()
        self.ignore_unrecongnized_events = False