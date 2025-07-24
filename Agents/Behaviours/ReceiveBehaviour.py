import codecs
import datetime
import pickle

from spade.behaviour import CyclicBehaviour
from spade.message import Message
from termcolor import colored

import Config

class ReceiveBehaviour(CyclicBehaviour):
    """
    Continuously listens for incoming weight messages.  When a message arrives:
      1) Logs the arrival.
      2) If the FSM is idle (not training), applies consensus at once to the backup model.
      3) Otherwise queues the message until training finishes.
      4) Always replies immediately with the current backup weights.
    """

    async def on_end(self):
        # Called when the entire FSM process stops
        print(f"[{self.agent.name}] ReceiveBehaviour finished")

    def _apply_consensus_to_backup(self, neighbour_msg):
        """
        Deserialize neighbour's weights, compute one-step consensus with backup_model,
        then update both the FederatedLearning helper and our backup model.
        """
        # 1) decode the neighbour's weight blob
        weights_blob, losses_blob, order_str = neighbour_msg.body.split("|")
        neighbour_weights = pickle.loads(codecs.decode(weights_blob.encode(), "base64"))

        # 2) ensure our max_order is at least as large
        neighbour_order = int(order_str)
        if self.agent.max_order < neighbour_order:
            self.agent.max_order = neighbour_order
            self.agent.epsilon_logger.write_to_file(str(self.agent.max_order))

        # 3) get our current backup and apply consensus
        local_backup = [self.agent.backup_model.state_dict()]
        eps = 1 / self.agent.max_order
        new_backup_dict = self.agent.consensus.apply_consensus(
            local_backup, neighbour_weights, eps
        )[0]

        # 4) update the backup model in place
        self.agent.backup_model.load_state_dict(new_backup_dict)

        # 5) tell the Federated helper about the new weights so that any
        #    future calls to .federated_learning use the updated backup
        self.agent.federated_learning.add_new_local_weight_local_losses(
            new_backup_dict, pickle.loads(codecs.decode(losses_blob.encode(), "base64"))
        )
        self.agent.federated_learning.set_model()

        print(colored(f"[{self.agent.name}] Applied consensus to backup model", 'red'))

    async def run(self):
        # 1) wait up to 4 seconds for a new message
        msg = await self.receive(timeout=4)
        if not msg:
            return  # no message this cycle

        # 2) log message receipt
        mid = msg.get_metadata("message_id")
        sender = msg.sender
        self.agent.message_logger.write_to_file(f"RECEIVE,{mid},{sender}")

        # 3) check message freshness
        now = datetime.datetime.now()
        sent_time = datetime.datetime.strptime(msg.get_metadata("timestamp"), "%Y-%m-%d %H:%M:%S.%f")
        if (now - sent_time).total_seconds() >= 3:
            # too old—ignore
            print(f"[{self.agent.name}] Ignored old message from {sender}")
            return

        # 4) record last message for possible FSM handling
        self.agent.last_message = msg

        # 5) prepare immediate response using backup_model weights
        response = Message(to=str(sender))
        # pack current backup weights, losses, and order
        weights_blob = self.agent.weights
        losses_blob = self.agent.losses
        order_str = str(self.agent.max_order)
        response.body = f"{weights_blob}|{losses_blob}|{order_str}"
        response.set_metadata("conversation", "response_data")
        response.set_metadata("timestamp", str(datetime.datetime.now()))
        response.set_metadata("message_id", mid)

        # 6) apply consensus now if FSM is not in training
        if self.agent.state_machine_behaviour.current_state != Config.TRAIN_STATE_AG:
            self._apply_consensus_to_backup(msg)
        else:
            # queue it for later in TrainState
            self.agent.pending_consensus_messages.append(msg)

        # 7) log and send the reply
        self.agent.message_logger.write_to_file(f"SEND_RESPONSE,{mid},{sender}")
        await self.send(response)