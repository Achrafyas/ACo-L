import copy
import pickle
import codecs
import datetime

from spade.agent import Agent
from spade.template import Template

from Agents import Config
from Agents.Behaviours.PresenceBehaviour import PresenceBehaviour
from Agents.Behaviours.ReceiveBehaviour import ReceiveBehaviour
from Agents.Behaviours.StateMachineBehaviour import StateMachineBehaviour
from Agents.Behaviours.StateMachineBehaviourStates.SetupState import SetupState
from Agents.Behaviours.StateMachineBehaviourStates.TrainState import TrainState
from Agents.Behaviours.StateMachineBehaviourStates.SendState import SendState
from Agents.Behaviours.StateMachineBehaviourStates.ReceiveState import ReceiveState
from Consensus.Consensus import Consensus
from Logs.Logger import Logger
from FederatedLearning.Federated import Federated


class FLAgent(Agent):
    """
    A federated‐learning agent running ACoL-style consensus in an FSM loop.
    Maintains a live model for training and a backup copy for responding
    to neighbours while training is in progress.
    """

    def __init__(
        self,
        jid: str,
        password: str,
        port: int,
        dataset: str,
        model_type: str,
        neighbours: list[str],
        model_path: str | None,
    ):
        super().__init__(jid, password)

        # 0) Coalition placeholders (required by launcher)
        self.coalition = []        # list of peers in my coalition
        self.coalition_index = -1  # index of my coalition, or -1 if none

        # --- Basic configuration ---
        self.port = port
        self.dataset = dataset
        self.neighbours = neighbours
        self.model_path = model_path

        # --- State machine & presence placeholders ---
        self.state_machine_behaviour = None
        self.presence_behaviour = None
        self.receive_behaviour = None

        # --- Tracking history & metrics for UI ---
        self.available_agents = []            # currently online neighbours
        self.message_history = []             # human‐readable log lines
        self.message_statistics = {}          # { agent_name: {"send": n, "receive": m}, ... }
        self.pending_consensus_messages = []
        self.message_statistics = {}
        # training/test curves
        self.train_accuracies = []
        self.train_losses    = []
        self.test_accuracies = []
        self.test_losses     = []

        # for computing epsilon = 1 / max_order
        self.max_order = 2

        # --- Consensus helper ---
        self.consensus = Consensus()
        # --- Build our live model via the Federated helper ---
        self.federated_learning = Federated(self.name, self.model_path, self.dataset, model_type)
        self.federated_learning.build_model()

        # live model: will be trained directly in TRAIN state
        self.model = self.federated_learning.get_model()
        # backup_model: snapshot we serve to neighbours during training
        self.backup_model = copy.deepcopy(self.model)

        # print model summary once
        self.federated_learning.print_model()

        # --- CSV loggers for later analysis ---
        self.weight_logger   = Logger(f"Logs/Weight Logs/{self.name}.csv",   Config.WEIGHT_LOGGER)
        self.training_logger = Logger(f"Logs/Training Logs/{self.name}.csv", Config.TRAINING_LOGGER)
        self.epsilon_logger  = Logger(f"Logs/Epsilon Logs/{self.name}.csv",  Config.EPSILON_LOGGER)
        self.message_logger  = Logger(f"Logs/Message Logs/{self.name}.csv",   Config.MESSAGE_LOGGER)
        self.training_time_logger = Logger(
            f"Logs/Training Time Logs/{self.name}.csv",
            Config.TRAINING_TIME_LOGGER,
        )

    async def setup(self):
        """
        Called once at startup.  Sets up:
          1) the web UI routes,
          2) the FSM behaviour,
          3) the Presence and Receive behaviours.
        """
        # --- Web dashboard for this agent ---
        self.web.add_get("/agent", self.agent_web_controller, "Agents/Interfaces/agent.html")
        self.web.add_post("/submit", self.agent_post_receiver, None)
        self.web.add_get("/agent/stop",  self.stop_agent, None)
        self.web.start(port=self.port)

        # --- FSM: SETUP → TRAIN → SEND → RECEIVE → TRAIN …
        self.state_machine_behaviour = StateMachineBehaviour()
        self.state_machine_behaviour.add_state(Config.SETUP_STATE_AG,   SetupState(),    initial=True)
        self.state_machine_behaviour.add_state(Config.TRAIN_STATE_AG,   TrainState())
        self.state_machine_behaviour.add_state(Config.SEND_STATE_AG,    SendState())
        self.state_machine_behaviour.add_state(Config.RECEIVE_STATE_AG, ReceiveState())

        self.state_machine_behaviour.add_transition(Config.SETUP_STATE_AG,   Config.TRAIN_STATE_AG)
        self.state_machine_behaviour.add_transition(Config.TRAIN_STATE_AG,   Config.SEND_STATE_AG)
        self.state_machine_behaviour.add_transition(Config.SEND_STATE_AG,    Config.RECEIVE_STATE_AG)
        self.state_machine_behaviour.add_transition(Config.RECEIVE_STATE_AG, Config.TRAIN_STATE_AG)
        self.state_machine_behaviour.add_transition(Config.SEND_STATE_AG, dest=Config.TRAIN_STATE_AG)


        fsm_template = Template()
        fsm_template.set_metadata("conversation", "response_data")
        self.add_behaviour(self.state_machine_behaviour, fsm_template)

        # --- Presence: track who’s online and subscribe ---
        self.presence_behaviour = PresenceBehaviour()
        self.add_behaviour(self.presence_behaviour)

        # --- ReceiveBehaviour: catch incoming consensus requests ---
        recv_template = Template()
        recv_template.set_metadata("conversation", "pre_consensus_data")
        self.receive_behaviour = ReceiveBehaviour()
        self.add_behaviour(self.receive_behaviour, recv_template)

    async def agent_web_controller(self, request):
        """
        Handler for GET /agent.  Supplies exactly the template context keys used
        in agent.html: epochs, curves, message_history, and message stats.
        """
        # Flatten history lines with newlines
        history_str = "\n".join(self.message_history)

        # Epoch indices 1…N
        epochs = list(range(1, len(self.test_accuracies) + 1))

        # Bare‐JIDs of neighbours
        agents = [jid.split("/")[0] for jid in self.available_agents]

        # Build send/receive counts aligned to `agents`
        recv_stats, send_stats = [], []
        for bare_jid in agents:
            name = bare_jid.split("@")[0]
            stats = self.message_statistics.get(name, {"send": 0, "receive": 0})
            recv_stats.append(stats["receive"])
            send_stats.append(stats["send"])

        return {
            "epochs":                    epochs,
            "test_accuracies":           self.test_accuracies,
            "train_accuracies":          self.train_accuracies,
            "test_losses":               self.test_losses,
            "train_losses":              self.train_losses,
            "received_message_statistics": recv_stats,
            "sent_message_statistics":     send_stats,
            "nb_available_agents":         len(agents),
            "available_agents":            agents,
            "message_history":             history_str,
        }

    async def agent_post_receiver(self, request):
        """
        Stub for POST /submit (if you need to handle form submissions on the agent UI).
        Currently just prints form data.
        """
        form = await request.post()
        print("Agent form data:", form)

    async def stop_agent(self, request):
        """
        Handler for GET /agent/stop: cleanly shut down all behaviours and the agent.
        """
        self.state_machine_behaviour.kill()
        self.presence_behaviour.kill()
        self.receive_behaviour.kill()
        await self.stop()