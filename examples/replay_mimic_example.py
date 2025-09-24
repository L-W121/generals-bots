# replay_mimic_example.py
from datasets import load_dataset
from generals.envs import PettingZooGenerals
from generals.agents import Agent

# ------------------------------
# 1️⃣ 加载 Hugging Face 数据集
# ------------------------------
print("Loading dataset...")
dataset = load_dataset("strakammm/generals_io_replays")
train_dataset = dataset['train']

# 选一场回放做示例
sample_replay = train_dataset[0]

# ------------------------------
# 2️⃣ 解析 replay 数据
# ------------------------------
def parse_replay(replay):
    moves = replay['moves']
    generals = replay['generals']
    cities = replay['cities']
    city_armies = replay['cityArmies']
    map_width = replay['mapWidth']
    map_height = replay['mapHeight']

    data = []
    for move in moves:
        player_idx, start, end, is_half, turn = move
        state = {
            'player_idx': player_idx,
            'turn': turn,
            'generals': generals,
            'cities': cities,
            'city_armies': city_armies,
            'map_width': map_width,
            'map_height': map_height
        }
        action = {
            'start': start,
            'end': end,
            'is_half': is_half
        }
        data.append((state, action))
    return data

parsed_data = parse_replay(sample_replay)

# ------------------------------
# 3️⃣ 创建 ReplayMimicAgent
# ------------------------------
class ReplayMimicAgent(Agent):
    def __init__(self, replay_data):
        super().__init__("MimicBot")
        self.replay_data = replay_data
        self.step = 0
        
    def reset(self):
        # 每次环境 reset 时，把计数器归零
        self.step = 0

    def act(self, observation):
        if self.step < len(self.replay_data):
            action = self.replay_data[self.step][1]  # 拿 moves
            self.step += 1
            # 简单处理：返回目标格子 end_tile
            return (0, action['start'], action['end'], 0, 0)  # (pass_turn, start, end, direction, extra)
        return (1, 0, 0, 0, 0)  # 超过步数就跳过

# ------------------------------
# 4️⃣ 创建环境并运行
# ------------------------------
    # 两个 agent：ReplayMimicAgent 和 RandomAgent
    from generals.agents import RandomAgent

    mimic_agent = ReplayMimicAgent(parsed_data)
    random_agent = RandomAgent()

    agents_dict = {mimic_agent.id: mimic_agent, random_agent.id: random_agent}
    agent_names = [mimic_agent.id, random_agent.id]

    env = PettingZooGenerals(agents=agent_names, render_mode="human")
    observations, info = env.reset()

    terminated = truncated = False
    while not (terminated or truncated):
        actions = {}
        for agent in env.agents:
            actions[agent] = agents_dict[agent].act(observations[agent])
        observations, rewards, terminated, truncated, info = env.step(actions)
        env.render()
