# standard
from functools import wraps
from inspect import isclass, signature
from typing import Callable, ParamSpec

# local
from src.pipeline.types import Condition, MetaState
from src.pipeline import Context


# step(context, *args, **kwargs) -> Context || step(self, context)
P = ParamSpec("P")
ParameterizedPipelineStep = Callable[P, Context]


def enforce(rules: dict[str, Condition | list[Condition]]):
    """
    Decorator factory that enforces variable state conditions on pipeline steps.

    Validates preconditions before step execution and postconditions after. Can be applied to both
    functions and classes (decorating their `__call__` method).
    """

    # unify to dict[str, list[Condition]]
    var_conditions = {
        variable: conditions if isinstance(conditions, list) else [conditions]
        for variable, conditions in rules.items()
    }

    def decorator(step: ParameterizedPipelineStep, fn_name_prefix: str = ""):
        # class PipelineStep: step.__call__(self, context, *args, **kwargs)
        if isclass(step):
            step.__call__ = decorator(step.__call__, fn_name_prefix=f"{step.__name__}.")
            return step
        
        # function PipelineStep: step(context, *args, **kwargs)
        fn_name = fn_name_prefix + step.__name__
        sig = signature(step)
        if "context" not in sig.parameters:
            raise TypeError(f"PipelineSteps must have a 'context' parameter")
        
        @wraps(step)
        def wrapper(*args, **kwargs):
            # unify signatures
            bound = sig.bind(*args, **kwargs)
            context = bound.arguments["context"]

            # preconditions
            var_bound_conditions = {}
            for variable, conditions in var_conditions.items():
                pre_metastate = _get_metastate(context, variable)
                bound_conditions = []
                for condition in conditions:
                    bound_condition = condition.bind(pre_metastate)
                    if not bound_condition:
                        raise RuntimeError(
                            f"{fn_name} failed precondition for variable '{variable}':"
                            f"\tcondition: {condition!r}"
                            f"\tpre-state: {pre_metastate}"
                        )
                    bound_conditions.append(bound_condition)
                var_bound_conditions[variable] = bound_conditions
                    
            result_context = step(*args, **kwargs)

            # postconditions
            for variable in var_bound_conditions.keys():
                post_metastate = _get_metastate(result_context, variable)
                conditions = var_conditions[variable]
                bound_conditions = var_bound_conditions[variable]

                for condition, bound_condition in zip(conditions, bound_conditions):
                    if not bound_condition.postcondition(post_metastate):
                        raise RuntimeError(
                            f"{fn_name} failed postcondition for variable '{variable}':"
                            f"\n\tcondition: {condition!r}"
                            f"\n\tpost-state: {post_metastate}"
                        )
            
            return result_context
        
        return wrapper
    
    return decorator

def _get_metastate(context: Context, key: str) -> MetaState:
    exists = key in context
    locked = False if not exists else context.is_locked(key)
    return MetaState(exists, locked)
