"""
Signal Store for managing shared data buses.

This module provides the SignalStore class, which acts as a shared workspace
for signals across the module graph. It decouples the signal flow from the
execution graph.
"""


from .module import Signal, SignalType


class SignalStore:
    """
    Shared workspace for signals in the synthesizer graph.

    The SignalStore allows modules to read from and write to shared keys (buses),
    enabling a more decoupled and flexible signal routing system.
    """

    def __init__(self):
        self._signals: dict[str, Signal] = {}

    def get(self, key: str, default: Signal | None = None) -> Signal:
        """
        Get a signal from the store by key.

        Args:
            key: The bus key to read from.
            default: The default signal to return if the key is not found.
                     If None, a default 0.0 AUDIO signal is returned.

        Returns:
            The signal associated with the key, or the default signal.
        """
        if default is None:
            default = Signal(SignalType.AUDIO, 0.0)
        return self._signals.get(key, default)

    def set(self, key: str, signal: Signal) -> None:
        """
        Set a signal in the store by key.

        Args:
            key: The bus key to write to.
            signal: The signal to store.
        """
        self._signals[key] = signal

    def clear(self) -> None:
        """Clear all signals from the store."""
        self._signals.clear()

    def get_all(self) -> dict[str, Signal]:
        """Get a copy of all signals in the store."""
        return self._signals.copy()
