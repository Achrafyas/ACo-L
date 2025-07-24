import codecs
import datetime
import pickle
import random
import uuid

from spade.behaviour import State, FSMBehaviour
from spade.message import Message
from termcolor import colored

import Config

class StateMachineBehaviour(FSMBehaviour):
    """
    Orchestrates the agent's main loop of:
      SETUP -> TRAIN -> SEND -> RECEIVE -> TRAIN ...
    by transitioning between named states and handling the data flow
    for each step of the ACoL protocol.
    """

    async def on_start(self):
        """
        Called once when the FSM begins.
        Logs the agent name and its initial state.
        """
        print(f"[{self.agent.name}] FSM starting in state {self.current_state}")

    async def on_end(self):
        """
        Called once when the FSM ends (if ever).
        Logs the final state before stopping.
        """
        print(f"[{self.agent.name}] FSM finished in state {self.current_state}")
        # Optionally, you could stop the agent here:
        # await self.agent.stop()