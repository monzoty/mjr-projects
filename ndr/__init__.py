"""NDR resolution system — NLP triage for failed-delivery customer replies."""

from .model import NDRResolver
from .normalize import preprocess, normalize
from .taxonomy import ACTIONS, INTENT_CODES, decide

__version__ = "0.1.0"
__all__ = ["NDRResolver", "preprocess", "normalize", "decide", "ACTIONS", "INTENT_CODES"]
