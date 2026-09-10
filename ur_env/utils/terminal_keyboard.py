import atexit
import os
import select
import signal
import sys
import termios
import threading
import tty
from typing import Callable


class TerminalKeyboardHub:
    """Shared terminal keyboard listener that works in WSL / plain terminals."""

    def __init__(self):
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._enabled = sys.stdin.isatty()
        self._old_settings = None
        self._fd = None
        self._owns_fd = False
        atexit.register(self.close)

    def register(self, callback: Callable[[str], None]):
        if not self._enabled:
            return lambda: None

        with self._lock:
            self._callbacks.append(callback)

        self._ensure_thread()

        def unregister():
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unregister

    def _ensure_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fd = None
        try:
            try:
                fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
                self._owns_fd = True
            except OSError:
                fd = sys.stdin.fileno()
                self._owns_fd = False

            self._fd = fd
            self._old_settings = termios.tcgetattr(fd)
            tty.setraw(fd)

            while not self._stop_event.is_set():
                rlist, _, _ = select.select([fd], [], [], 0.1)
                if not rlist:
                    continue
                try:
                    ch = os.read(fd, 1)
                except OSError:
                    continue
                if not ch:
                    continue
                if isinstance(ch, bytes):
                    # Ctrl+C (0x03) or Ctrl+D (0x04) → send SIGINT to self
                    if ch in (b'\x03', b'\x04'):
                        # Restore terminal before signal
                        if self._old_settings is not None:
                            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
                            self._old_settings = None
                        os.kill(os.getpid(), signal.SIGINT)
                        break
                    ch = ch.decode(errors="ignore")
                with self._lock:
                    callbacks = list(self._callbacks)
                for callback in callbacks:
                    try:
                        callback(ch)
                    except Exception:
                        # Keep other callbacks alive even if one consumer fails.
                        pass
        except Exception:
            # If raw terminal mode cannot be established, degrade silently.
            pass
        finally:
            if self._old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
                except Exception:
                    pass
            if self._fd is not None and self._owns_fd:
                try:
                    os.close(self._fd)
                except Exception:
                    pass
                self._fd = None

    def close(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def suspend(self):
        """Temporarily restore normal terminal mode (e.g. for tqdm/printf)."""
        if self._old_settings is not None and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def resume(self):
        """Re-enter raw mode after suspend()."""
        if self._fd is not None:
            try:
                tty.setraw(self._fd)
            except Exception:
                pass


TERMINAL_KEYBOARD_HUB = TerminalKeyboardHub()
