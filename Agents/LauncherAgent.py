from spade.agent import Agent
from pathlib import Path
import networkx as nx
import shutil

import time
import aiohttp_jinja2
import threading
import asyncio
import Config
from Agents.FLAgent import FLAgent


class LauncherAgent(Agent):

    def __init__(self, jid: str, password: str, port, agent_port):
        super().__init__(jid, password)
        self.create_required_folders()
        self.neighbors_list = []
        self.port = port
        self.agent_port = agent_port
        self.agent = None
        self.nx_graph = None
        self.agents = []
        self.threads = []
        self.started = False

    def wait_for_agents(agents):
        alive = True
        while alive:
            for entity in agents:
                if entity.is_alive():
                    alive = True
                    break
                alive = False
            time.sleep(1)

    def create_required_folders(self) -> None:
        for folder in Config.logs_folders:
            Path(f"{Config.logs_root_folder}/{folder}").mkdir(
                parents=True, exist_ok=True
            )
        Path("Saved Models").mkdir(parents=True, exist_ok=True)

    async def load_graph(self, graph_file, node_id):
        """
        Load the GraphML file and store the neighbours of the agent
        :param graph_file: GraphML
        :param node_id: agent ID in the graph
        """
        self.nx_graph = nx.read_graphml(graph_file.file)
        self.neighbors_list = list(self.nx_graph.neighbors(node_id))

    def load_model_file(self, model_file):
        """
        Load the file containing the model that the user uploaded and save it locally
        :param model_file: model file
        :return: path of the locally saved model
        """
        local_path = "Saved Models/uploaded_model.pt"
        with open(local_path, "wb", encoding="utf-8") as file:
            local_file = file.read()
        shutil.copyfileobj(model_file.file, local_file)
        return local_path

    @aiohttp_jinja2.template("Agents/Interfaces/launcher.html")
    async def agent_web_controller(self, request):
        return {
        # context for your template
        "foo": "bar",
        # …
        }   

    def launch_agent(agent):
        # each thread needs its own event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # start() is a coroutine
        loop.run_until_complete(agent.start(auto_register=True))
        # now keep the loop alive so SPADE can run its behaviours
        try:
            loop.run_forever()
        finally:
            loop.close()


    async def agent_post_receiver(self, request):
        """
        Handles the form when submitted by the user in the launcher graphical interface.
        """
        form = await request.post()
        print(form)
        if form["agentCreationMethod"] == "graph":
            print("Graph")
            node_id = form["nodeId_graph"]
            graph_file_data = form["graphInputFile"]
            model_file_data = form["modelFile_graph"]
            model_path = None
            if hasattr(model_file_data, "file"):
                model_path = self.load_model_file(model_file_data)
            if form["datasetSelection_graph"] == "mnist_graph":
                dataset = "mnist"
            elif form["datasetSelection_graph"] == "fmnist_graph":
                dataset = "fmnist"
            elif form["datasetSelection_graph"] == "cifar4_graph":
                dataset = "cifar4_coal"
            elif form["datasetSelection_graph"] == "fruit4_graph":
                dataset = "fruit4"
            elif form["datasetSelection_graph"] == "cifar8_graph":
                dataset = "cifar8"

            if form["modelSelection_graph"] == "mlp_graph":
                model_type = "mlp"
            else:
                model_type = "cnn"
            print(model_path)
            port = form["port_graph"]
            # await self.load_graph(graph_file_data, node_id)
            self.nx_graph = nx.read_graphml(graph_file_data.file)
           # instantiate one FLAgent per node in the GML
            self.agents = []
            for node_id in self.nx_graph.nodes():
                neighbours = list(self.nx_graph.neighbors(node_id))
                print(f"[{node_id}] neighbours:", neighbours)
                agent_port = str(int(port) + len(self.agents))
                agent_jid = f"{node_id}@{Config.xmpp_server}"
                ag = FLAgent(
                    agent_jid,
                    "abcdefg",
                    agent_port,
                    dataset,
                    model_type,
                    neighbours,
                    model_path,
                )
                self.agents.append(ag)

            print(
                f"Coalition information:\n Probability: {Config.coalition_probability:.2f}"
            )
            for agent in self.agents:
                print(
                    f"[{agent.name}] belongs to {agent.coalition_index} with {agent.coalition}"
                )
            await asyncio.sleep(10)
            print("\nLaunching agents...")
            for agent in self.agents:
                t = threading.Thread(
                    target=LauncherAgent.launch_agent,
                    args=[agent],
                )
                t.daemon = True
                t.name = agent.name
                self.threads.append(t)
                t.start()
                print(f"Agent {agent.name} launched.")
            self.started = True
        else:
            print("No graph")
            node_id = form["nodeId_no_graph"]
            neighbours = form["agent_neighbours_no_graph"].split(",")
            model_file_data = form["modelFile_no_graph"]
            model_path = None
            if hasattr(model_file_data, "file"):
                model_path = self.load_model_file(model_file_data)
            if form["datasetSelection_no_graph"] == "mnist_no_graph":
                dataset = "mnist"
            else:
                dataset = "fmnist"
            if form["modelSelection_no_graph"] == "mlp_no_graph":
                model_type = "mlp"
            else:
                model_type = "cnn"
            port = form["port_no_graph"]

            self.agent = FLAgent(
                node_id + "@" + Config.xmpp_server,
                "abcdefg",
                port,
                dataset,
                model_type,
                neighbours,
                model_path,
            )
            await self.agent.start(auto_register=True)

    async def setup(self):
        print(f"Launcher agent set in: http://{Config.url}:{self.port}/agent")
        self.web.add_get(
            "/agent", self.agent_web_controller, "Agents/Interfaces/launcher.html"
        )
        self.web.add_post("/submit", self.agent_post_receiver, None)
        self.web.start(port=self.port)