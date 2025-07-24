from spade.behaviour import OneShotBehaviour
from termcolor import colored
import Config

class PresenceBehaviour(OneShotBehaviour):
    """
    Keeps track of which neighbours are online and manages subscriptions.
    """

    async def run(self):
        # Attach presence event handlers
        self.presence.on_subscribe   = self.on_subscribe
        self.presence.on_subscribed  = self.on_subscribed
        self.presence.on_available   = self.on_available
        self.presence.on_unavailable = self.on_unavailable

        # Announce ourselves as available
        self.presence.set_available()

        # Subscribe to each neighbour from the experiment graph
        for neighbour in self.agent.neighbours:
            jid = f"{neighbour}@{Config.xmpp_server}"
            self.presence.subscribe(jid)

    def on_subscribe(self, jid, *args):
        # A neighbour wants to subscribe to our presence
        print(colored(f"[{self.agent.name}] {jid.split('@')[0]} requested subscription → approving", 'green'))
        # Add to our available list if not already present
        if jid not in self.agent.available_agents:
            self.agent.available_agents.append(jid)
        # Also subscribe back so we can send/receive messages
        self.presence.subscribe(jid)

    def on_subscribed(self, jid, *args):
        # They accepted our subscription request
        print(colored(f"[{self.agent.name}] {jid.split('@')[0]} accepted our subscription", 'green'))

    def on_available(self, jid, stanza, *args):
        # A neighbour came online
        name = jid.split("@")[0]
        print(colored(f"[{self.agent.name}] {name} is now available", 'green'))

    def on_unavailable(self, jid, stanza, *args):
        # A neighbour went offline
        name = jid.split("@")[0]
        print(colored(f"[{self.agent.name}] {name} is now unavailable", 'green'))

        # Remove them from our list and update max_order for epsilon
        if jid in self.agent.available_agents:
            self.agent.available_agents.remove(jid)

        # max_order = number of active neighbours
        self.agent.max_order = len(self.agent.available_agents)
        # Log the new epsilon-related value (just count for now)
        self.agent.epsilon_logger.write_to_file(str(self.agent.max_order))
        print(f"Max order updated to: {self.agent.max_order}")

    async def on_end(self):
        # Called once this behaviour completes
        print("PresenceBehaviour finished")