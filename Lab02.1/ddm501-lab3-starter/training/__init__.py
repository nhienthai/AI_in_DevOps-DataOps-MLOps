"""
Training-time helpers for the Movie Rating Prediction system.

This package is intentionally kept out of ``app/``: it is only needed to build
a model artifact, never to serve one. The single exception is unpickling - a
model produced by the ``local`` backend references :mod:`training.local_svd`,
so the package must stay importable wherever such an artifact is loaded.
"""

__all__ = ["dataset", "local_svd"]
