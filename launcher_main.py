import asyncio
import time
from Agents.LauncherAgent import LauncherAgent
import Config

async def main():
    # instantiate your launcher
    n0 = LauncherAgent("my_launcher_agent@localhost", "abcdefg", 10000, 11000)
    # await its start()
    await n0.start(auto_register=True)

    # now keep the script alive while the agent is running
    try:
        while n0.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await n0.stop()

if __name__ == "__main__":
    asyncio.run(main())
