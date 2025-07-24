import codecs
import pickle
import random

from spade.behaviour import State
from termcolor import colored

import Config

class TrainState(State):
    """
    TRAIN state: perform one round of local training,
    then process any queued consensus messages,
    and finally transition to SEND.
    """

    async def run(self):
        # 1) Log entering TRAIN and mark start time
        print(f"[{self.agent.name}] TRAINING")
        self.agent.training_time_logger.write_to_file("START")

        # 2) Snapshot the current live model into backup_model
        #    so any incoming consensus requests during training
        #    see the last fully trained copy.
        self.agent.backup_model.load_state_dict(self.agent.model.state_dict())

        # 3) Train the live model (self.agent.model) for one epoch
        training_id = random.randint(0, 1000)
        print(f"Training ID: {training_id}")
        (local_weights, local_losses,
         self.agent.train_acc, self.agent.train_loss,
         self.agent.test_acc, self.agent.test_loss
        ) = await self.agent.federated_learning.train_local_model()

        # 4) Log training metrics (accuracy, loss)
        print(f"Completed training {training_id}")
        self.agent.training_logger.write_to_file(
            f"{self.agent.train_acc},"
            f"{self.agent.train_loss},"
            f"{self.agent.test_acc},"
            f"{self.agent.test_loss}"
        )

        # 5) Serialize new weights & losses for later consensus
        self.agent.weights = codecs.encode(pickle.dumps(local_weights), "base64").decode()
        self.agent.losses  = codecs.encode(pickle.dumps(local_losses),  "base64").decode()

        # 6) Commit snapshot: copy freshly trained live model → backup_model
        self.agent.backup_model.load_state_dict(self.agent.model.state_dict())

        # 7) Mark end of training time
        self.agent.training_time_logger.write_to_file("STOP")

        # 8) Save metrics to agent history for UI
        self.agent.test_accuracies.append(self.agent.test_acc)
        self.agent.test_losses.append(self.agent.test_loss)
        self.agent.train_accuracies.append(self.agent.train_acc)
        self.agent.train_losses.append(self.agent.train_loss)

        # 9) If any consensus requests arrived during training, apply them now
        while self.agent.pending_consensus_messages:
            msg = self.agent.pending_consensus_messages.pop(0)
            # Delegate to ReceiveBehaviour helper to merge into backup_model
            await self.agent.receive_behaviour._apply_consensus_to_backup(msg)

        # 10) Transition to SEND state
        self.set_next_state(Config.SEND_STATE_AG)
