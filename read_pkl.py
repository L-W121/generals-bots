# 验证样本格式
import torch
samples = torch.load("data/cache/preprocessed/batch_000000_000100.pt")
state_tensor, action_id, metadata = samples[0]

print(f"状态形状: {state_tensor.shape}")  # 应该是 [15, H, W]
print(f"动作ID: {action_id}")             # 整数
print(f"元数据: {metadata.keys()}")       # replay_id, turn, player_id 等
