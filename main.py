import time

from Agents.FLAgent import FLAgent
from Agents.LauncherAgent import LauncherAgent
import Config

n0 = FLAgent("age0@localhost", "abcdefg", 20000, "mnist", "cnn", ["age1"], None)
f1 = n0.start(auto_register=True)
f1.result()

while n0.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        n0.stop()
        break