"""Explanation generation (Grad-CAM, Grad-CAM++, Integrated Gradients,
SHAP, LIME) and quantitative explanation validation (heatmap-mask IoU,
pointing-game accuracy, false-focus rate, shuffled-attribution baseline,
top-k SHAP ablation faithfulness). Explanation quality is asserted with
metrics here, never judged by whether a heatmap "looks right."
"""

from .base import BaseExplainer
