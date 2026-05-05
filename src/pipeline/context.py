# standard
from copy import deepcopy
from typing import Any, Self

class Context:
    """
    A context object that manages variables with locking capabilities.

    Provides a dictionary-like interface for storing and retrieving variables with additional
    locking functionality to prevent unintended modification or deletion during pipeline execution.
    """

    __slots__ = ("_vars", "_locked")

    def __init__(self):
        self._vars = {}
        self._locked = set()
    
    # === minimal dictionary interface ===

    def __getitem__(self, key: str) -> Any:
        self._validate_key(key, must_exist=True)
        return self._vars[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._validate_key(key, must_exist=False)
        if key in self._locked:
            raise ValueError(f"'{key}' is locked and cannot be modified")
        self._vars[key] = value  # handles definition and modification
    
    def __delitem__(self, key: str):
        self._validate_key(key, must_exist=True)
        if key in self._locked:
            raise ValueError(f"'{key}' is locked and cannot be deleted")
        del self._vars[key]
        # key already not in self._locked

    def __contains__(self, key: str) -> bool:
        return key in self._vars
    
    def __len__(self) -> int:
        return len(self._vars)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._vars.get(key, default)
    
    # === locking/unlocking

    def lock(self, key: str, strict: bool = True):
        self._validate_key(key, must_exist=True)
        if key in self._locked and strict:
            raise ValueError(f"'{key}' is already locked")
        self._locked.add(key)
    
    def unlock(self, key: str, strict: bool = True):
        self._validate_key(key, must_exist=True)
        if key not in self._locked and strict:
            raise ValueError(f"'{key}' already unlocked")
        self._locked.discard(key)

    def is_locked(self, key: str) -> bool:
        self._validate_key(key, must_exist=True)
        return key in self._locked
    
    # === misc ===

    def copy(self, deep: bool = True) -> Self:
        new_ = Context()
        new_._vars = deepcopy(self._vars) if deep else self._vars.copy()
        new_._locked = set(self._locked)  # strings are immutable
        return new_
    
    def _validate_key(self, key: str, must_exist: bool):
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}: {key!r}")
        if must_exist and key not in self._vars:
            raise KeyError(f"Unknown key '{key}'")
