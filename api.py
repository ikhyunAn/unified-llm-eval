# api.py
"""
Public convenience API for the unified_llm_eval package.

This module centralizes the "convenience layer" that callers (e.g. the
merge_tools perturbation-sweep hook) previously reimplemented on their side:

  * TASK_ALIASES       - map abstract task names to registry keys.
  * default_env_config - build an env_config with vendor paths resolved against
                         the package root (absolute), independent of CWD.
  * evaluate_model     - run a single (model, task) evaluation and return a
                         numeric score, hiding the evaluator/registry plumbing.

Importing this package no longer requires a sys.path hack or an
UNIFIED_LLM_EVAL_ROOT env var: vendor paths are derived from this file's
location.
"""

import os
from pathlib import Path

from .utils.env_manager import EnvironmentManager
from .tasks.task_registry import TASK_REGISTRY

# Package root == this file's directory (the repo becomes the package).
PACKAGE_ROOT = Path(__file__).parent.resolve()

# Abstract task name -> registry key. Callers may pass either the alias
# ("coder", "math") or the registry key ("humaneval", "gsm8k-cot") directly.
TASK_ALIASES = {
    "coder": "humaneval",
    "math": "gsm8k-cot",
}


def default_env_config():
    """
    Return the environment_config dict used to drive the evaluators, with all
    vendored tool directories resolved to absolute paths under the package root.

    The conda environment names ("harness_env", "languages_env", "goedelv2")
    match the convention used throughout the experiment YAML configs.
    """
    return {
        "harness_env": "harness_env",
        "languages_env": "languages_env",
        "goedelv2_env": "goedelv2",
        "math_harness_dir": str(PACKAGE_ROOT / "vendor" / "math-evaluation-harness"),
        "language_eval_dir": str(PACKAGE_ROOT / "vendor" / "HumanEval"),
        "goedel_prover_dir": str(PACKAGE_ROOT / "vendor" / "goedel-prover"),
    }


def resolve_task_name(task_name):
    """Resolve an abstract task alias to its registry key (identity if unknown)."""
    return TASK_ALIASES.get(task_name, task_name)


def evaluate_model(
    model_path,
    task_name,
    env_manager=None,
    env_config=None,
    eval_settings=None,
    run_id=1,
):
    """
    Evaluate a single model on a single task and return the numeric score.

    Args:
        model_path: Local directory path or a HuggingFace Hub model ID.
        task_name: Abstract alias ("coder"/"math") or a registry key.
        env_manager: Optional EnvironmentManager. Created from env_config if None.
        env_config: Optional env_config dict. Defaults to default_env_config().
        eval_settings: Optional eval settings dict (gpu_ids, temperature, etc.).
        run_id: Run identifier passed through to the evaluator.

    Returns:
        float: The parsed score (percentage), or 0.0 on failure.
    """
    if env_config is None:
        env_config = default_env_config()
    if eval_settings is None:
        eval_settings = {}
    if env_manager is None:
        env_manager = EnvironmentManager(
            env_config.get("harness_env"),
            env_config.get("languages_env"),
            env_config.get("goedelv2_env", "goedelv2"),
        )

    registry_task_name = resolve_task_name(task_name)
    task_info = TASK_REGISTRY.get(registry_task_name)
    if not task_info:
        print(f"Task '{registry_task_name}' not in registry.")
        return 0.0

    EvaluatorClass = task_info["evaluator"]
    evaluator = EvaluatorClass(env_manager, env_config, eval_settings)

    # If model_path is an existing local directory, resolve to an absolute path.
    # Otherwise treat it as a HuggingFace Hub ID and pass it through as-is --
    # resolving a Hub ID like "org/model" with Path.resolve() would produce a
    # nonexistent local path that lm_eval rejects.
    if os.path.isdir(model_path):
        resolved_path = str(Path(model_path).resolve())
        model_name = os.path.basename(resolved_path)
    else:
        resolved_path = model_path
        model_name = model_path.split("/")[-1]

    model_config = {
        "model_name": model_name,
        "path": resolved_path,
        "location": "local",
    }

    result_dict = evaluator.evaluate(model_config, registry_task_name, run_id=run_id)
    if result_dict.get("status") == "SUCCESS":
        try:
            score_val = result_dict["score"]
            if isinstance(score_val, str):
                return float(score_val.replace("%", ""))
            return float(score_val)
        except Exception:
            return 0.0
    print(f"Evaluation failed: {result_dict.get('error_log', 'Unknown error')}")
    return 0.0
