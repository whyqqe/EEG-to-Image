"""Model components: the shared EEG trunk and the SAMCLIP assembly.

There is no subject-conditioning module here any more. See
``samclip.models.subject_conditioning`` for why it was removed -- the file is kept as
a module-level tombstone (documentation only, no code and no importers) so the
decision and its measurements stay in the tree.
"""
from .backbone import EEGTrunk, EncoderLayer, TemporalSpatialAggregator
from .samclip import SAMCLIP, LayerRouter, build_model

__all__ = [
    "EEGTrunk", "EncoderLayer", "TemporalSpatialAggregator",
    "SAMCLIP", "LayerRouter", "build_model",
]
