#  type: ignore
import gymnasium as gym
import numpy as np

from generals.envs import GymnasiumGenerals
from generals import GridFactory

agent_names = ["007", "Generalissimo"]

grid_factory = GridFactory(
    min_grid_dims=(24, 24),
    max_grid_dims=(24, 24),
)

n_envs = 4
envs = gym.vector.AsyncVectorEnv(
    [lambda: GymnasiumGenerals(agents=agent_names, grid_factory=grid_factory, truncation=500) for _ in range(n_envs)],
)


observations, infos = envs.reset()
terminated = [False] * len(observations)
truncated = [False] * len(observations)

import time
start_time = time.time()
for i in range(40000):
    agent_actions = [envs.single_action_space.sample() for _ in range(n_envs)]
    npc_actions = [envs.single_action_space.sample() for _ in range(n_envs)]

    # Stack actions together
    actions = np.stack([agent_actions, npc_actions], axis=1)
    observations, rewards, terminated, truncated, infos = envs.step(actions)
    masks = [np.stack([infos[agent_name]["masks"] for agent_name in agent_names])]
    # if any(terminated) or any(truncated):
    #     break
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
