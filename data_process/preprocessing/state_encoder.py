"""
Generals.io 状态编码器
将游戏状态转换为 15 通道张量表示
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import logging

class StateEncoder:
    """将游戏状态编码为多通道张量"""
    
    def __init__(self, num_channels: int = 15):
        """
        初始化状态编码器
        
        Args:
            num_channels: 输出通道数
        """
        self.num_channels = num_channels
        self.logger = logging.getLogger(__name__)
        
        # 游戏常量
        self.MOUNTAIN = -1
        self.EMPTY = 0
        self.CITY = -2
        
        # 通道定义
        self.channel_names = [
            "my_army",          # 0: 己方军队数量
            "enemy_army",       # 1: 敌方军队数量  
            "my_territory",     # 2: 己方控制区域
            "enemy_territory",  # 3: 敌方控制区域
            "mountains",        # 4: 山脉/障碍物
            "cities",           # 5: 城市位置
            "generals",         # 6: 将军位置
            "visible_area",     # 7: 可见区域
            "fog_of_war",       # 8: 雾战区域
            "growth_potential", # 9: 军队增长潜力
            "territory_age",    # 10: 领土控制时间
            "borders",          # 11: 边界信息
            "attack_potential", # 12: 攻击潜力
            "defend_priority",  # 13: 防御优先级
            "strategic_value"   # 14: 战略价值
        ]
    
    def encode_state(self, 
                    game_state: Dict, 
                    player_id: int,
                    normalize: bool = True) -> torch.Tensor:
        """
        将游戏状态编码为张量
        
        Args:
            game_state: 游戏状态字典
            player_id: 当前玩家ID (0 或 1)
            normalize: 是否标准化数值
            
        Returns:
            形状为 [num_channels, height, width] 的张量
        """
        height, width = game_state['height'], game_state['width']
        channels = []
        
        try:
            # 通道 0: 己方军队数量
            my_army = self._encode_my_army(game_state, player_id)
            channels.append(my_army)
            
            # 通道 1: 敌方军队数量
            enemy_army = self._encode_enemy_army(game_state, player_id)
            channels.append(enemy_army)
            
            # 通道 2: 己方控制区域
            my_territory = self._encode_my_territory(game_state, player_id)
            channels.append(my_territory)
            
            # 通道 3: 敌方控制区域
            enemy_territory = self._encode_enemy_territory(game_state, player_id)
            channels.append(enemy_territory)
            
            # 通道 4: 山脉/障碍物
            mountains = self._encode_mountains(game_state)
            channels.append(mountains)
            
            # 通道 5: 城市位置
            cities = self._encode_cities(game_state)
            channels.append(cities)
            
            # 通道 6: 将军位置
            generals = self._encode_generals(game_state, player_id)
            channels.append(generals)
            
            # 通道 7: 可见区域
            visible = self._encode_visibility(game_state, player_id)
            channels.append(visible)
            
            # 通道 8: 雾战区域
            fog = self._encode_fog_of_war(game_state, player_id)
            channels.append(fog)
            
            # 通道 9: 军队增长潜力
            growth = self._encode_growth_potential(game_state, player_id)
            channels.append(growth)
            
            # 通道 10: 领土控制时间
            territory_age = self._encode_territory_age(game_state, player_id)
            channels.append(territory_age)
            
            # 通道 11: 边界信息
            borders = self._encode_borders(game_state, player_id)
            channels.append(borders)
            
            # 通道 12: 攻击潜力
            attack_potential = self._encode_attack_potential(game_state, player_id)
            channels.append(attack_potential)
            
            # 通道 13: 防御优先级
            defend_priority = self._encode_defend_priority(game_state, player_id)
            channels.append(defend_priority)
            
            # 通道 14: 战略价值
            strategic_value = self._encode_strategic_value(game_state, player_id)
            channels.append(strategic_value)
            
            # 堆叠所有通道
            state_tensor = np.stack(channels, axis=0)
            
            # 标准化
            if normalize:
                state_tensor = self._normalize_channels(state_tensor)
            
            return torch.FloatTensor(state_tensor)
            
        except Exception as e:
            self.logger.error(f"状态编码失败: {e}")
            # 返回零张量作为后备
            return torch.zeros(self.num_channels, height, width)
    
    def _encode_my_army(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码己方军队数量"""
        height, width = game_state['height'], game_state['width']
        my_army = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        armies = game_state['armies']
        
        # 己方领土的军队数
        my_mask = (terrain == player_id + 1)
        my_army[my_mask] = armies[my_mask]
        
        return my_army
    
    def _encode_enemy_army(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码敌方军队数量"""
        height, width = game_state['height'], game_state['width']
        enemy_army = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        armies = game_state['armies']
        
        # 敌方领土的军队数（所有非己方玩家）
        enemy_mask = (terrain > 0) & (terrain != player_id + 1)
        enemy_army[enemy_mask] = armies[enemy_mask]
        
        return enemy_army
    
    def _encode_my_territory(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码己方控制区域"""
        height, width = game_state['height'], game_state['width']
        my_territory = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        my_territory[terrain == player_id + 1] = 1.0
        
        return my_territory
    
    def _encode_enemy_territory(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码敌方控制区域"""
        height, width = game_state['height'], game_state['width']
        enemy_territory = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        enemy_mask = (terrain > 0) & (terrain != player_id + 1)
        enemy_territory[enemy_mask] = 1.0
        
        return enemy_territory
    
    def _encode_mountains(self, game_state: Dict) -> np.ndarray:
        """编码山脉/障碍物"""
        height, width = game_state['height'], game_state['width']
        mountains = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        mountains[terrain == self.MOUNTAIN] = 1.0
        
        return mountains
    
    def _encode_cities(self, game_state: Dict) -> np.ndarray:
        """编码城市位置"""
        height, width = game_state['height'], game_state['width']
        cities = np.zeros((height, width), dtype=np.float32)
        
        # 从城市列表获取位置
        for pos in game_state['cities']:
            row, col = self._pos_to_coord(pos, width)
            if 0 <= row < height and 0 <= col < width:
                cities[row, col] = 1.0
        
        return cities
    
    def _encode_generals(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码将军位置"""
        height, width = game_state['height'], game_state['width']
        generals = np.zeros((height, width), dtype=np.float32)
        
        # 己方将军 +1，敌方将军 -1
        for i, pos in enumerate(game_state['generals']):
            row, col = self._pos_to_coord(pos, width)
            if 0 <= row < height and 0 <= col < width:
                if i == player_id:
                    generals[row, col] = 1.0
                else:
                    generals[row, col] = -1.0
        
        return generals
    
    def _encode_visibility(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码可见区域（简化版本）"""
        height, width = game_state['height'], game_state['width']
        visible = np.zeros((height, width), dtype=np.float32)
        
        # 简化：己方控制区域及其相邻区域可见
        my_territory = (game_state['terrain'] == player_id + 1)
        
        # 扩展可见区域到相邻位置
        for i in range(height):
            for j in range(width):
                if my_territory[i, j]:
                    # 标记当前位置和相邻位置为可见
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            visible[ni, nj] = 1.0
        
        return visible
    
    def _encode_fog_of_war(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码雾战区域"""
        visible = self._encode_visibility(game_state, player_id)
        fog = 1.0 - visible  # 不可见区域为雾战
        return fog
    
    def _encode_growth_potential(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码军队增长潜力"""
        height, width = game_state['height'], game_state['width']
        growth = np.zeros((height, width), dtype=np.float32)
        
        # 己方城市和领土每回合都会增长
        terrain = game_state['terrain']
        my_territory = (terrain == player_id + 1)
        growth[my_territory] = 1.0
        
        # 城市增长更快
        for pos in game_state['cities']:
            row, col = self._pos_to_coord(pos, width)
            if 0 <= row < height and 0 <= col < width and my_territory[row, col]:
                growth[row, col] = 2.0
        
        return growth
    
    def _encode_territory_age(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码领土控制时间（简化版本）"""
        height, width = game_state['height'], game_state['width']
        # 简化：使用当前回合数作为近似
        territory_age = np.zeros((height, width), dtype=np.float32)
        
        my_territory = (game_state['terrain'] == player_id + 1)
        territory_age[my_territory] = min(game_state['turn'] / 100.0, 1.0)
        
        return territory_age
    
    def _encode_borders(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码边界信息"""
        height, width = game_state['height'], game_state['width']
        borders = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        my_territory = (terrain == player_id + 1)
        
        # 找到边界位置（己方领土与非己方领土相邻）
        for i in range(height):
            for j in range(width):
                if my_territory[i, j]:
                    # 检查相邻位置
                    is_border = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if not my_territory[ni, nj]:
                                is_border = True
                                break
                    
                    if is_border:
                        borders[i, j] = 1.0
        
        return borders
    
    def _encode_attack_potential(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码攻击潜力"""
        height, width = game_state['height'], game_state['width']
        attack_potential = np.zeros((height, width), dtype=np.float32)
        
        terrain = game_state['terrain']
        armies = game_state['armies']
        my_territory = (terrain == player_id + 1)
        
        # 边界位置的攻击潜力基于军队数量
        for i in range(height):
            for j in range(width):
                if my_territory[i, j] and armies[i, j] > 1:
                    # 检查是否可以攻击相邻敌方位置
                    can_attack = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_owner = terrain[ni, nj]
                            if (neighbor_owner != player_id + 1 and 
                                neighbor_owner != self.MOUNTAIN):
                                can_attack = True
                                break
                    
                    if can_attack:
                        attack_potential[i, j] = min(armies[i, j] / 10.0, 1.0)
        
        return attack_potential
    
    def _encode_defend_priority(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码防御优先级"""
        height, width = game_state['height'], game_state['width']
        defend_priority = np.zeros((height, width), dtype=np.float32)
        
        # 将军位置最高优先级
        general_pos = game_state['generals'][player_id]
        general_row, general_col = self._pos_to_coord(general_pos, width)
        if 0 <= general_row < height and 0 <= general_col < width:
            defend_priority[general_row, general_col] = 1.0
        
        # 城市也有较高防御优先级
        terrain = game_state['terrain']
        my_territory = (terrain == player_id + 1)
        
        for pos in game_state['cities']:
            row, col = self._pos_to_coord(pos, width)
            if 0 <= row < height and 0 <= col < width and my_territory[row, col]:
                defend_priority[row, col] = 0.7
        
        return defend_priority
    
    def _encode_strategic_value(self, game_state: Dict, player_id: int) -> np.ndarray:
        """编码战略价值"""
        height, width = game_state['height'], game_state['width']
        strategic_value = np.zeros((height, width), dtype=np.float32)
        
        # 综合考虑多个因素
        cities = self._encode_cities(game_state)
        generals = self._encode_generals(game_state, player_id)
        borders = self._encode_borders(game_state, player_id)
        
        # 城市价值高
        strategic_value += cities * 0.8
        
        # 将军位置价值最高
        strategic_value += np.abs(generals) * 1.0
        
        # 边界位置有一定价值
        strategic_value += borders * 0.3
        
        # 限制在 [0, 1] 范围内
        strategic_value = np.clip(strategic_value, 0, 1)
        
        return strategic_value
    
    def _normalize_channels(self, state_tensor: np.ndarray) -> np.ndarray:
        """标准化通道数值"""
        normalized = state_tensor.copy()
        
        # 对军队数量通道进行对数标准化
        for channel_idx in [0, 1]:  # 己方和敌方军队
            channel = normalized[channel_idx]
            if channel.max() > 0:
                # 对数变换 + 标准化到 [0, 1]
                channel_log = np.log1p(channel)
                if channel_log.max() > 0:
                    normalized[channel_idx] = channel_log / channel_log.max()
        
        # 其他通道已经在 [0, 1] 或 [-1, 1] 范围内
        return normalized
    
    def _pos_to_coord(self, pos: int, width: int) -> Tuple[int, int]:
        """将一维位置转换为二维坐标"""
        row = pos // width
        col = pos % width
        return row, col
    
    def get_channel_info(self) -> Dict:
        """获取通道信息"""
        return {
            'num_channels': self.num_channels,
            'channel_names': self.channel_names,
            'channel_descriptions': {
                'my_army': '己方军队数量分布',
                'enemy_army': '敌方军队数量分布',
                'my_territory': '己方控制区域',
                'enemy_territory': '敌方控制区域',
                'mountains': '山脉和障碍物位置',
                'cities': '城市位置',
                'generals': '将军位置（己方+1，敌方-1）',
                'visible_area': '可见区域',
                'fog_of_war': '雾战区域',
                'growth_potential': '军队增长潜力',
                'territory_age': '领土控制时间',
                'borders': '领土边界',
                'attack_potential': '攻击潜力',
                'defend_priority': '防御优先级',
                'strategic_value': '综合战略价值'
            }
        }