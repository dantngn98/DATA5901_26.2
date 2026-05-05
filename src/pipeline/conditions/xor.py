# local
from src.pipeline.types import BoundCondition, Condition, MetaState

class XOR(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = list(conditions)
    
    def bind(self, state: MetaState) -> BoundCondition | None:
        bound_conditions = []
        for condition in self.conditions:
            bound = condition.bind(state)
            if bound:
                bound_conditions.append(bound)
        return bound_conditions[0] if len(bound_conditions) == 1 else None
    
    def __repr__(self) -> str:
        return f"XOR({",".join([cond.__repr__() for cond in self.conditions])})"
