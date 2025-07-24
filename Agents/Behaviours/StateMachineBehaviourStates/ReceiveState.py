import codecs
import datetime
import pickle

from spade.behaviour import State
from termcolor import colored

import Config

class ReceiveState(State):
    """
    RECEIVE state: after sending our weights, wait for the neighbour's response,
    then apply consensus on the backup model and transition back to TRAIN.
    """

    def _apply_consensus_to_backup(self, msg):
        """
        Deserialize a reply message, apply one consensus step on the backup_model,
        update federated helper, and re-serialize for future sends.
        """
        # 1) Deserialize neighbour's weights, losses and their max_order
        weights_blob, losses_blob, order_str = msg.body.split("|")
        neighbour_weights = pickle.loads(codecs.decode(weights_blob.encode(), "base64"))
        neighbour_losses  = pickle.loads(codecs.decode(losses_blob.encode(),  "base64"))
        neighbour_order   = int(order_str)

        # 2) Update max_order if neighbour's is larger
        if self.agent.max_order < neighbour_order:
            self.agent.max_order = neighbour_order

        # 3) Consensus on backup snapshot
        local_snapshot = [ self.agent.backup_model.state_dict() ]
        eps = 1 / self.agent.max_order
        updated = self.agent.consensus.apply_consensus(local_snapshot, neighbour_weights, eps)
        new_backup_state = updated[0]

        # 4) Load updated state into backup_model, update federated helper
        self.agent.backup_model.load_state_dict(new_backup_state)
        self.agent.federated_learning.add_new_local_weight_local_losses(new_backup_state, neighbour_losses)
        self.agent.federated_learning.set_model()

        # 5) Re-serialize for next send
        self.agent.weights = codecs.encode(pickle.dumps([new_backup_state]), "base64").decode()
        self.agent.losses  = codecs.encode(pickle.dumps(neighbour_losses),   "base64").decode()

    async def run(self):
        print(f"[{self.agent.name}] RECEIVE")
        # 1) Wait up to 10s for reply
        msg = await self.receive(timeout=10)

        if not msg:
            # 2a) No reply → log and go back to TRAIN
            print(f"[{self.agent.name}] No response received → retrain")
            self.set_next_state(Config.TRAIN_STATE_AG)
            return

        # 2b) Log receipt
        now = datetime.datetime.now()
        sender_jid = str(msg.sender).split("/")[0]
        self.agent.message_history.insert(
            0, f"{now.hour}:{now.minute}:{now.second} : Received response from {sender_jid}"
        )
        print(colored(f"[{self.agent.name}] Response from {sender_jid}", 'cyan'))
        mid = msg.get_metadata("message_id")
        self.agent.message_logger.write_to_file(f"RECEIVE_RESPONSE,{mid},{sender_jid}")
        stats = self.agent.message_statistics.setdefault(
            sender_jid.split("@")[0], {"send": 0, "receive": 0}
        )
        stats["receive"] += 1

        # 3) Apply consensus using the new helper
        self._apply_consensus_to_backup(msg)
        print(colored(f"[{self.agent.name}] Applied consensus after response", 'red'))

        # 4) Back to TRAIN
        self.set_next_state(Config.TRAIN_STATE_AG)