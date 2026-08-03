"""PyRadiomics feature extraction from segmentation masks (first-order,
shape, GLCM, GLRLM, GLSZM, GLDM), z-scoring on the training cohort only,
and the variance/mutual-information pre-filter down to the QUBO budget.
"""

from .base import BaseExtractor
