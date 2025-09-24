"""
Generals.io 动作编码器
负责动作与网络输出之间的编码/解码转换
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
import logging

class ActionEncoder:
    """动作编码器，处理游戏动作的编码和解码"""
    
    def __init__(self, map_width: int, map_height: int):
        """
        初始化动作编码器
        
        Args:
            map_width: 地图宽度
            map_height: 地图高度
        """
        self.width = map_width
        self.height = map_height
        self.total_positions = map_width * map_height
        self.logger = logging.getLogger(__name__)
        
        # 方向定义 (row_offset, col_offset)
        self.directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
            4: (0, 0)    # pass/不移动
        }
        
        # 反向方向映射
        self.direction_to_id = {v: k for k, v in self.directions.items()}
        
        # 动作空间大小
        # 方案1: from_pos * total_positions + to_pos
        self.action_space_size_v1 = self.total_positions * self.total_positions
        
        # 方案2: from_pos * 5 + direction (更紧凑)  
        self.action_space_size_v2 = self.total_positions * 5
        
        # 默认使用方案2（更紧凑）
        self.action_space_size = self.action_space_size_v2
        self.encoding_method = "direction_based"  # "position_pair" 或 "direction_based"
    
    def encode_move(self, move: List[int], method: Optional[str] = None) -> int:
        """
        编码单个移动
        
        Args:
            move: [player, from_pos, to_pos, is_half, turn]
            method: 编码方法 ("position_pair" 或 "direction_based")
            
        Returns:
            编码后的动作ID
        """
        if method is None:
            method = self.encoding_method
            
        player, from_pos, to_pos, is_half, turn = move
        
        if method == "position_pair":
            return self._encode_position_pair(from_pos, to_pos, is_half)
        elif method == "direction_based":
            return self._encode_direction_based(from_pos, to_pos, is_half)
        else:
            raise ValueError(f"未知的编码方法: {method}")
    
    def decode_action(self, action_id: int, method: Optional[str] = None) -> Tuple[int, int, bool]:
        """
        解码动作ID为具体移动
        
        Args:
            action_id: 动作ID
            method: 解码方法
            
        Returns:
            (from_pos, to_pos, is_half)
        """
        if method is None:
            method = self.encoding_method
            
        if method == "position_pair":
            return self._decode_position_pair(action_id)
        elif method == "direction_based":
            return self._decode_direction_based(action_id)
        else:
            raise ValueError(f"未知的解码方法: {method}")
    
    def _encode_position_pair(self, from_pos: int, to_pos: int, is_half: bool) -> int:
        """位置对编码方法"""
        # 简化版本：忽略 is_half，只编码位置对
        # 更完整版本可以扩展动作空间来包含 is_half
        return from_pos * self.total_positions + to_pos
    
    def _decode_position_pair(self, action_id: int) -> Tuple[int, int, bool]:
        """位置对解码方法"""
        from_pos = action_id // self.total_positions
        to_pos = action_id % self.total_positions
        is_half = False  # 简化版本
        return from_pos, to_pos, is_half
    
    def _encode_direction_based(self, from_pos: int, to_pos: int, is_half: bool) -> int:
        """基于方向的编码方法"""
        # 计算方向
        direction = self._get_direction(from_pos, to_pos)
        
        # 编码: from_pos * 5 + direction
        # 注意：这里忽略了 is_half，可以通过扩展动作空间来支持
        base_action = from_pos * 5 + direction
        
        # 简化版本：不处理 is_half
        return base_action
    
    def _decode_direction_based(self, action_id: int) -> Tuple[int, int, bool]:
        """基于方向的解码方法"""
        from_pos = action_id // 5
        direction_id = action_id % 5
        
        # 获取方向偏移
        if direction_id in self.directions:
            direction_offset = self.directions[direction_id]
            to_pos = self._apply_direction(from_pos, direction_offset)
        else:
            to_pos = from_pos  # pass 动作
        
        is_half = False  # 简化版本
        return from_pos, to_pos, is_half
    
    def _get_direction(self, from_pos: int, to_pos: int) -> int:
        """计算从 from_pos 到 to_pos 的方向ID"""
        from_row, from_col = self._pos_to_coord(from_pos)
        to_row, to_col = self._pos_to_coord(to_pos)
        
        # 计算偏移
        row_offset = to_row - from_row
        col_offset = to_col - from_col
        
        # 查找对应的方向ID
        direction_offset = (row_offset, col_offset)
        
        if direction_offset in self.direction_to_id:
            return self.direction_to_id[direction_offset]
        else:
            # 非法移动，返回 pass 动作
            return 4
    
    def _apply_direction(self, from_pos: int, direction_offset: Tuple[int, int]) -> int:
        """在指定位置应用方向偏移"""
        from_row, from_col = self._pos_to_coord(from_pos)
        row_offset, col_offset = direction_offset
        
        to_row = from_row + row_offset
        to_col = from_col + col_offset
        
        # 检查边界
        if 0 <= to_row < self.height and 0 <= to_col < self.width:
            return self._coord_to_pos(to_row, to_col)
        else:
            return from_pos  # 越界，返回原位置
    
    def encode_legal_actions(self, game_state: Dict, player_id: int) -> List[int]:
        """
        编码所有合法动作
        
        Args:
            game_state: 游戏状态
            player_id: 玩家ID
            
        Returns:
            合法动作ID列表
        """
        legal_actions = []
        terrain = game_state['terrain']
        armies = game_state['armies']
        
        # 遍历所有己方控制的位置
        for pos in range(self.total_positions):
            row, col = self._pos_to_coord(pos)
            
            # 检查是否为己方领土且有足够军队
            if (terrain[row, col] == player_id + 1 and 
                armies[row, col] > 1):
                
                # 检查所有可能的方向
                for direction_id, direction_offset in self.directions.items():
                    to_row = row + direction_offset[0]
                    to_col = col + direction_offset[1]
                    
                    # 检查目标位置是否合法
                    if self._is_valid_target(game_state, to_row, to_col):
                        if self.encoding_method == "direction_based":
                            action_id = pos * 5 + direction_id
                        else:
                            to_pos = self._coord_to_pos(to_row, to_col)
                            action_id = pos * self.total_positions + to_pos
                        
                        legal_actions.append(action_id)
        
        return legal_actions
    
    def create_action_mask(self, game_state: Dict, player_id: int) -> torch.Tensor:
        """
        创建动作掩码张量
        
        Args:
            game_state: 游戏状态
            player_id: 玩家ID
            
        Returns:
            动作掩码张量，合法动作为1，非法动作为0
        """
        mask = torch.zeros(self.action_space_size, dtype=torch.bool)
        legal_actions = self.encode_legal_actions(game_state, player_id)
        
        if legal_actions:
            mask[legal_actions] = True
        
        return mask
    
    def _is_valid_target(self, game_state: Dict, row: int, col: int) -> bool:
        """检查目标位置是否合法"""
        height, width = game_state['height'], game_state['width']
        
        # 检查边界
        if not (0 <= row < height and 0 <= col < width):
            return False
        
        # 检查是否为山脉
        terrain = game_state['terrain']
        if terrain[row, col] == -1:  # MOUNTAIN
            return False
        
        return True
    
    def _pos_to_coord(self, pos: int) -> Tuple[int, int]:
        """将一维位置转换为二维坐标"""
        row = pos // self.width
        col = pos % self.width
        return row, col
    
    def _coord_to_pos(self, row: int, col: int) -> int:
        """将二维坐标转换为一维位置"""
        return row * self.width + col
    
    def batch_encode_moves(self, moves: List[List[int]]) -> torch.Tensor:
        """
        批量编码移动
        
        Args:
            moves: 移动列表
            
        Returns:
            编码后的动作张量
        """
        action_ids = []
        for move in moves:
            try:
                action_id = self.encode_move(move)
                action_ids.append(action_id)
            except Exception as e:
                self.logger.warning(f"编码移动失败 {move}: {e}")
                action_ids.append(0)  # 默认为第一个动作
        
        return torch.LongTensor(action_ids)
    
    def apply_action_mask(self, action_logits: torch.Tensor, 
                         action_mask: torch.Tensor) -> torch.Tensor:
        """
        应用动作掩码，将非法动作的logits设为负无穷
        
        Args:
            action_logits: 动作logits张量
            action_mask: 动作掩码张量
            
        Returns:
            掩码后的logits
        """
        masked_logits = action_logits.clone()
        masked_logits[~action_mask] = float('-inf')
        return masked_logits
    
    def sample_action(self, action_logits: torch.Tensor, 
                     action_mask: torch.Tensor,
                     temperature: float = 1.0,
                     method: str = "multinomial") -> int:
        """
        从动作分布中采样动作
        
        Args:
            action_logits: 动作logits
            action_mask: 合法动作掩码
            temperature: 温度参数
            method: 采样方法 ("greedy", "multinomial", "top_k")
            
        Returns:
            采样的动作ID
        """
        # 应用掩码
        masked_logits = self.apply_action_mask(action_logits, action_mask)
        
        # 应用温度
        if temperature != 1.0:
            masked_logits = masked_logits / temperature
        
        if method == "greedy":
            return torch.argmax(masked_logits).item()
        elif method == "multinomial":
            probs = torch.softmax(masked_logits, dim=0)
            return torch.multinomial(probs, 1).item()
        else:
            raise ValueError(f"未知的采样方法: {method}")
    
    def get_action_info(self) -> Dict:
        """获取动作编码器信息"""
        return {
            'map_size': (self.height, self.width),
            'total_positions': self.total_positions,
            'action_space_size': self.action_space_size,
            'encoding_method': self.encoding_method,
            'directions': self.directions,
            'action_space_sizes': {
                'position_pair': self.action_space_size_v1,
                'direction_based': self.action_space_size_v2
            }
        }

class ExtendedActionEncoder(ActionEncoder):
    """扩展的动作编码器，支持半军队移动"""
    
    def __init__(self, map_width: int, map_height: int):
        super().__init__(map_width, map_height)
        
        # 扩展动作空间来支持半军队移动
        # 每个位置-方向组合有两个版本：全军队和半军队
        self.action_space_size = self.total_positions * 5 * 2
        self.supports_half_move = True
    
    def encode_move(self, move: List[int], method: Optional[str] = None) -> int:
        """编码移动，支持半军队移动"""
        player, from_pos, to_pos, is_half, turn = move
        
        # 获取基础动作ID
        direction = self._get_direction(from_pos, to_pos)
        base_action = from_pos * 5 + direction
        
        # 扩展以支持半军队移动
        if is_half:
            action_id = base_action + self.total_positions * 5
        else:
            action_id = base_action
        
        return action_id
    
    def decode_action(self, action_id: int, method: Optional[str] = None) -> Tuple[int, int, bool]:
        """解码动作，支持半军队移动"""
        # 检查是否为半军队移动
        is_half = action_id >= (self.total_positions * 5)
        
        if is_half:
            base_action = action_id - (self.total_positions * 5)
        else:
            base_action = action_id
        
        # 解码基础动作
        from_pos = base_action // 5
        direction_id = base_action % 5
        
        direction_offset = self.directions[direction_id]
        to_pos = self._apply_direction(from_pos, direction_offset)
        
        return from_pos, to_pos, is_half