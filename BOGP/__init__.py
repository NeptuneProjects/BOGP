from ax.storage.botorch_modular_registry import ACQUISITION_FUNCTION_REGISTRY
from ax.storage.botorch_modular_registry import REVERSE_ACQUISITION_FUNCTION_REGISTRY
from botorch.acquisition import (
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    ProbabilityOfImprovement,
)

ACQUISITION_FUNCTION_REGISTRY.update(
    {
        ProbabilityOfImprovement: "ProbabilityOfImprovement",
        qProbabilityOfImprovement: "qProbabilityOfImprovement",
        qUpperConfidenceBound: "qUpperConfidenceBound",
    }
)
REVERSE_ACQUISITION_FUNCTION_REGISTRY.update(
    {
        "ProbabilityOfImprovement": ProbabilityOfImprovement,
        "qProbabilityOfImprovement": qProbabilityOfImprovement,
        "qUpperConfidenceBound": qUpperConfidenceBound,
    }
)
