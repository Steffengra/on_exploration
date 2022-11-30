
from numpy import (
    round,
    argmin,
)


# TODO: This scheduler is hard coded for a certain config
class ManualScheduler:
    def __init__(
            self,
            rng
    ) -> None:
        self.rng = rng

    def get_action(
            self,
            state,
    ) -> int:
        step_id = round(state[0]) * 6
        puncture_queue_indicator = state[1]
        critical_indicator = state[2]
        occupied_lengths = state[3:]

        if step_id < 5:
            return 0
        else:
            if puncture_queue_indicator == 1:
                if any(occupied_lengths == 0):
                    return argmin(occupied_lengths) + 1
                else:
                    return 1
            else:
                return 0
