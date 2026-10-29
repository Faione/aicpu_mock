"""Structured, value-free lifecycle diagnostics."""

from contextlib import contextmanager
import logging
import os

from .model import canonical


def event(name, **details):
    logging.getLogger("qfa_mock").info("qfa_mock %s", canonical({"event": name, "pid": os.getpid(), **details}))


@contextmanager
def phase(name, **details):
    event("stage_start", stage=name, **details)
    try:
        yield
    except BaseException as error:
        if isinstance(error, Exception) and not hasattr(error, "stage"):
            error.stage = name
        event("stage_failed", stage=name, error_type=type(error).__name__,
              code=getattr(error, "code", None), **details)
        raise
    else:
        event("stage_complete", stage=name, **details)
