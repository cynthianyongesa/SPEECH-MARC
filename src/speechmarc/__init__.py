"""
SPEECH-MARC: Speech-based Multimodal Assessment of Risk for Cognition

This package provides tools for extracting linguistic and acoustic features,
residualizing them against covariates, and training machine learning models 
to detect Mild Cognitive Impairment (MCI).
"""

__version__ = "0.1.0"
__all__ = ["preprocessing", "features", "residualize", "models", "evaluate"]

from . import preprocessing, features, residualize, models, evaluate

__author__ = "Cynthia Nyongesa"
__license__ = "MIT"

