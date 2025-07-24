import datetime
import random
import uuid
import pickle
import codecs

from spade.behaviour import State
from spade.message import Message

import Config

class SendState(State):
    """
    SEND state: pick one active neighbour at random and send
    our current backup_model snapshot so they can perform consensus.
    """

    def send_message(self, recipient):
        # 1) Create the XMPP message
        msg = Message(to=recipient)
        msg.set_metadata("conversation", "pre_consensus_data")
        msg.set_metadata("timestamp", str(datetime.datetime.now()))

        # 2) If we have no backup yet, send a placeholder
        if not hasattr(self.agent, "backup_model") or self.agent.backup_model is None:
            msg.body = "None"
        else:
            # 3) Serialize just the backup_model’s weights
            snapshot = [self.agent.backup_model.state_dict()]
            weights_blob = codecs.encode(pickle.dumps(snapshot), "base64").decode()
            # losses we kept in self.agent.losses already refers to last snapshot
            losses_blob = self.agent.losses or "None"
            order_str = str(round(self.agent.max_order, 3))

            # 4) Build the multipart body
            msg.body = f"{weights_blob}|{losses_blob}|{order_str}"
            print(f"Message length: {len(msg.body)}")

        return msg

    async def run(self):
        # 1) If no neighbours, skip sending and go back to TRAIN
        if not self.agent.available_agents:
            self.set_next_state(Config.TRAIN_STATE_AG)
            return

        # 2) Pick a random neighbour
        neighbour_jid = random.choice(self.agent.available_agents).split("/")[0]
        print(f"[{self.agent.name}] SEND to {neighbour_jid}")

        # 3) Build the message
        msg = self.send_message(neighbour_jid)
        message_id = str(uuid.uuid4())
        msg.set_metadata("message_id", message_id)

        # 4) Log send stats
        name = neighbour_jid.split("@")[0]
        stats = self.agent.message_statistics.setdefault(name, {"send": 0, "receive": 0})
        stats["send"] += 1
        self.agent.message_logger.write_to_file(f"SEND,{message_id},{name}")

        # 5) Actually send
        await self.send(msg)

        # 6) Record in UI history
        now = datetime.datetime.now()
        self.agent.message_history.insert(
            0, f"{now.hour}:{now.minute}:{now.second} : Sent to {name}"
        )

        # 7) Transition to RECEIVE to wait for reply
        self.set_next_state(Config.RECEIVE_STATE_AG)
