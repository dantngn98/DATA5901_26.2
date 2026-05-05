# local
from src.pipeline.types import BoundCondition, Condition, MetaState

class Sequence(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = conditions
    
    def bind(self, state: MetaState) -> BoundCondition | None:
        for condition in self.conditions[:-1]:
            # precondition check
            bound_condition = condition.bind(state)
            if not bound_condition:
                return None
            
            # simulate state into next condition in sequence
            state = bound_condition.simulate()
            if not state:
                return None
        
        return self.conditions[-1].bind(state)
    
    def __repr__(self) -> str:
        return f"Sequence({",".join([cond.__repr__() for cond in self.conditions])})"
