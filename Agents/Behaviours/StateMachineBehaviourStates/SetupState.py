from spade.behaviour import State
import Config

class SetupState(State):
    """
    Initial state: perform any setup needed before training.
    Immediately transitions into the TRAIN state.
    """

    async def run(self):
        # Log entry into SETUP
        print(f"[{self.agent.name}] State: SETUP")
        # Move straight to training in the next cycle
        self.set_next_state(Config.TRAIN_STATE_AG)