"""
unified_llm_eval - convenience/control layer around several evaluation backends
(lm-evaluation-harness, math-evaluation-harness, a multi-language code harness,
and Goedel-Prover-V2 for Lean 4 theorem proving).

Public API. Import as a package (the repository directory must be named
``unified_llm_eval`` on the consumer's sys.path):

    from unified_llm_eval import (
        EnvironmentManager, TASK_REGISTRY, ALL_TASKS,
        evaluate_model, default_env_config, TASK_ALIASES,
    )

No sys.path insertion or UNIFIED_LLM_EVAL_ROOT env var is required; vendored
tool paths are resolved relative to this package (see api.default_env_config).
"""

from .utils.env_manager import EnvironmentManager
from .tasks.task_registry import TASK_REGISTRY, ALL_TASKS
from .api import (
    evaluate_model,
    default_env_config,
    resolve_task_name,
    TASK_ALIASES,
)

__all__ = [
    "EnvironmentManager",
    "TASK_REGISTRY",
    "ALL_TASKS",
    "evaluate_model",
    "default_env_config",
    "resolve_task_name",
    "TASK_ALIASES",
]
