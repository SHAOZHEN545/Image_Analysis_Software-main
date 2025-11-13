'''Directory monitor for incoming image files.'''

import time
import sys, os
from watchdog.observers import Observer
from watchforchange import *
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from watchdog.events import PatternMatchingEventHandler
from watchdog.events import DirCreatedEvent, DirDeletedEvent, DirModifiedEvent, DirMovedEvent


class Monitor():
    """Watch a directory for changes and trigger a callback."""

    def __init__(self, path, func, fileSize):
        """Initialize the monitor.

        Parameters
        ----------
        path : str
            Directory path to watch.
        func : callable
            Callback executed when a change is detected.
        fileSize : float
            Expected file size in megabytes.
        """
        self.filePath = path
        self.fileSize = fileSize
        self.func = MyHandler(func, self.fileSize)
        self.oldObserver = None
        self.observer = None
        self._running = False
        

    def oldObserver(self):
        """Return the previous Observer instance."""
        return self.oldObserver
        
    def createObserverAndStart(self):
        """Create and start the watchdog observer."""

        if self.observer is not None:
            # Ensure any existing observer is fully shut down before creating
            # a new instance. This mirrors the behaviour expected by the
            # watchdog observer API and prevents duplicate threads.
            self.stop()
            self.join()

        self.observer = Observer()
        self.observer.schedule(self.func, self.filePath)
        self.observer.start()
        self._running = True

    def join(self):
        """Block until the observer thread terminates."""
        if self.observer is not None:
            self.observer.join()
            self.observer = None
            
    def stop(self):
        """Stop the observer."""
        if self.observer is not None and self._running:
            self.observer.stop()
            self._running = False

    def is_running(self):
        """Return True when the observer thread is active."""

        return self.observer is not None and self._running

    def pause(self):
        """Temporarily stop watching without discarding configuration."""

        if not self.is_running():
            return False
        self.stop()
        self.join()
        return True

    def resume(self):
        """Resume watching after a pause."""

        if self.is_running():
            return False
        self.createObserverAndStart()
        return True
        
        
        
    def changeFilePath(self, newFilePath):
        """Change the directory being monitored."""

        self.stop()
        self.join()
        self.oldObserver = None
        self.filePath = newFilePath
        self.createObserverAndStart()
