'''Utility to watch directories for changes.'''

import time
import sys, os

try:
    import wx
except ImportError:  # pragma: no cover - wx is optional during tests
    wx = None

from watchdog.observers import Observer
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff
from watchdog.events import PatternMatchingEventHandler
from watchdog.events import DirCreatedEvent, DirDeletedEvent, DirModifiedEvent, DirMovedEvent


class MyHandler(PatternMatchingEventHandler):
    """Event handler that waits for a file to finish writing before firing."""

    def __init__(self, functionToCall, arg):
        super().__init__()
        self.f = functionToCall
        try:
            expected_mb = float(arg)
        except (TypeError, ValueError):
            expected_mb = 0
        # ``expectedFileSize`` is used as an upper bound when known (e.g. Andor
        # cameras). When the camera produces smaller files, fall back to a
        # stability check so the callback is triggered as soon as the file
        # write settles.
        self.expectedFileSize = max(0, expected_mb) * 1024 ** 2
        self.lastModTime = -1
        self._retry_delay = 0.05
        self._stable_checks_required = 3
        self._max_checks = 40

    def _trigger_callback(self):
        """Invoke the callback on the UI thread when wx is available."""

        if wx is None:
            self.f()
            return

        try:
            if wx.IsMainThread():
                self.f()
            else:
                wx.CallAfter(self.f)
        except Exception as exc:
            # Logging instead of raising prevents watchdog from stopping if the
            # UI thread is no longer available.
            print(f"Failed to dispatch directory change callback: {exc}")

    def _wait_for_file(self, path):
        """Wait for the file write to stabilise before processing.

        The handler now considers both the expected file size (when known) and
        the observed growth of the file. This avoids a fixed delay for smaller
        cameras while still giving slower writers a brief window to finish.
        """

        if not self._retry_delay:
            return

        stable_polls = 0
        last_size = -1

        for _ in range(self._max_checks):
            try:
                statinfo = os.stat(path)
            except FileNotFoundError:
                filesize = 0
            else:
                filesize = int(statinfo.st_size)

            if self.expectedFileSize and filesize >= self.expectedFileSize:
                break

            if filesize > 0 and filesize == last_size:
                stable_polls += 1
                if stable_polls >= self._stable_checks_required:
                    break
            else:
                stable_polls = 0

            last_size = filesize
            time.sleep(self._retry_delay)

    def process(self, event):
        if getattr(event, "is_directory", False):
            return
        self._wait_for_file(event.src_path)
        self._trigger_callback()
    def on_created(self, event):
        self.process(event)
		
if __name__ == '__main__':
    args = sys.argv[1:]
    observer = Observer()
    observer.schedule(MyHandler(), path = args[0] if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(1)
            print("tick tock------")
    except KeyboardInterrupt:
        observer.stop()
        observer.join()