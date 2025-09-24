"""
Generals.io 对局回放解析器
负责从 Hugging Face 数据集加载和解析专家对局数据
"""

from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from pathlib import Path

class GeneralsReplayParser:
    """解析 Generals.io 对局回放数据"""
    
    def __init__(self, 
                 dataset_name: str = "strakammm/generals_io_replays",
                 cache_dir: Optional[str] = None):
        """
        初始化回放解析器
        
        Args:
            dataset_name: Hugging Face 数据集名称
            cache_dir: 缓存目录路径
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.dataset = None
        self.logger = logging.getLogger(__name__)
        
        # 游戏常量
        self.MOUNTAIN = -1
        self.EMPTY = 0
        self.CITY = -2
        
    def load_dataset(self) -> None:
        """加载数据集"""
        try:
            self.logger.info(f"正在加载数据集: {self.dataset_name}")
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir
            )
            self.logger.info(f"数据集加载完成，包含 {len(self.dataset['train'])} 个对局")
        except Exception as e:
            self.logger.error(f"数据集加载失败: {e}")
            raise
    
    def get_dataset_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if self.dataset is None:
            self.load_dataset()
            
        train_data = self.dataset['train']
        stats = {
            'total_replays': len(train_data),
            'map_sizes': {},
            'total_moves': 0,
            'avg_moves_per_game': 0,
            'players': set(),
            'avg_game_length': 0
        }
        
        # 分析前1000个对局（避免全量分析耗时过长）
        sample_size = min(1000, len(train_data))
        total_moves = 0
        total_turns = 0
        
        for i in range(sample_size):
            replay = train_data[i]
            
            # 地图尺寸统计
            map_size = (replay['mapWidth'], replay['mapHeight'])
            stats['map_sizes'][map_size] = stats['map_sizes'].get(map_size, 0) + 1
            
            # 移动统计
            moves_count = len(replay['moves'])
            total_moves += moves_count
            
            # 游戏长度统计（最后一个移动的回合数）
            if replay['moves']:
                last_turn = replay['moves'][-1][4]  # turn 是第5个元素
                total_turns += last_turn
            
            # 玩家统计
            for username in replay['usernames']:
                stats['players'].add(username)
        
        stats['total_moves'] = total_moves
        stats['avg_moves_per_game'] = total_moves / sample_size
        stats['avg_game_length'] = total_turns / sample_size
        stats['unique_players'] = len(stats['players'])
        
        return stats
    
    def parse_replay(self, replay_data: Dict) -> List[Dict]:
        """
        解析单个对局回放
        
        Args:
            replay_data: 对局数据字典
            
        Returns:
            游戏状态序列列表
        """
        width = replay_data['mapWidth']
        height = replay_data['mapHeight']
        moves = replay_data['moves']
        
        # 初始化游戏状态
        initial_state = self._initialize_game_state(replay_data)
        
        # 重建游戏状态序列
        game_states = [initial_state.copy()]
        current_state = initial_state.copy()
        
        for move in moves:
            # 应用移动并更新状态
            self._apply_move(current_state, move)
            game_states.append(current_state.copy())
        
        return game_states
    
    def _initialize_game_state(self, replay_data: Dict) -> Dict:
        """初始化游戏状态"""
        width = replay_data['mapWidth']
        height = replay_data['mapHeight']
        
        # 初始化地图
        terrain = np.zeros((height, width), dtype=np.int32)
        armies = np.zeros((height, width), dtype=np.int32)
        
        # 设置山脉
        for pos in replay_data['mountains']:
            row, col = self._pos_to_coord(pos, width)
            terrain[row, col] = self.MOUNTAIN
        
        # 设置城市
        for i, pos in enumerate(replay_data['cities']):
            row, col = self._pos_to_coord(pos, width)
            terrain[row, col] = self.CITY
            armies[row, col] = replay_data['cityArmies'][i]
        
        # 设置将军位置
        for player_id, pos in enumerate(replay_data['generals']):
            row, col = self._pos_to_coord(pos, width)
            terrain[row, col] = player_id + 1  # 玩家ID从1开始
            armies[row, col] = 1  # 将军初始军队数
        
        return {
            'width': width,
            'height': height,
            'terrain': terrain,
            'armies': armies,
            'turn': 0,
            'usernames': replay_data['usernames'],
            'stars': replay_data['stars'],
            'mountains': set(replay_data['mountains']),
            'cities': replay_data['cities'],
            'generals': replay_data['generals']
        }
    
    def _apply_move(self, game_state: Dict, move: List) -> None:
        """
        应用移动到游戏状态
        
        Args:
            game_state: 当前游戏状态
            move: [player, from_pos, to_pos, is_half, turn]
        """
        player, from_pos, to_pos, is_half, turn = move
        width = game_state['width']
        
        # 更新回合数
        game_state['turn'] = turn
        
        # 转换位置为坐标
        from_row, from_col = self._pos_to_coord(from_pos, width)
        to_row, to_col = self._pos_to_coord(to_pos, width)
        
        # 检查移动合法性
        if not self._is_valid_move(game_state, from_row, from_col, to_row, to_col, player):
            return
        
        # 计算移动的军队数
        from_armies = game_state['armies'][from_row, from_col]
        if is_half:
            move_armies = from_armies // 2
        else:
            move_armies = from_armies - 1  # 保留1个军队在原位置
        
        if move_armies <= 0:
            return
        
        # 执行移动
        target_armies = game_state['armies'][to_row, to_col]
        target_owner = game_state['terrain'][to_row, to_col]
        
        if target_owner == player + 1:  # 移动到己方领土
            game_state['armies'][to_row, to_col] += move_armies
        elif target_owner == 0 or target_owner == self.CITY:  # 占领空地或城市
            if move_armies > target_armies:
                game_state['terrain'][to_row, to_col] = player + 1
                game_state['armies'][to_row, to_col] = move_armies - target_armies
            else:
                game_state['armies'][to_row, to_col] = target_armies - move_armies
        elif target_owner > 0 and target_owner != player + 1:  # 攻击敌方
            if move_armies > target_armies:
                game_state['terrain'][to_row, to_col] = player + 1
                game_state['armies'][to_row, to_col] = move_armies - target_armies
            else:
                game_state['armies'][to_row, to_col] = target_armies - move_armies
        
        # 更新原位置
        game_state['armies'][from_row, from_col] -= move_armies
    
    def _is_valid_move(self, game_state: Dict, from_row: int, from_col: int, 
                      to_row: int, to_col: int, player: int) -> bool:
        """检查移动是否合法"""
        height, width = game_state['height'], game_state['width']
        
        # 检查坐标范围
        if not (0 <= from_row < height and 0 <= from_col < width and
                0 <= to_row < height and 0 <= to_col < width):
            return False
        
        # 检查是否相邻
        if abs(from_row - to_row) + abs(from_col - to_col) != 1:
            return False
        
        # 检查起始位置是否属于当前玩家
        if game_state['terrain'][from_row, from_col] != player + 1:
            return False
        
        # 检查起始位置是否有足够军队
        if game_state['armies'][from_row, from_col] <= 1:
            return False
        
        # 检查目标位置不是山脉
        if game_state['terrain'][to_row, to_col] == self.MOUNTAIN:
            return False
        
        return True
    
    def _pos_to_coord(self, pos: int, width: int) -> Tuple[int, int]:
        """将一维位置转换为二维坐标"""
        row = pos // width
        col = pos % width
        return row, col
    
    def _coord_to_pos(self, row: int, col: int, width: int) -> int:
        """将二维坐标转换为一维位置"""
        return row * width + col
    
    def export_preprocessed_data(self, output_dir: str, max_replays: Optional[int] = None):
        """
        导出预处理后的数据
        
        Args:
            output_dir: 输出目录
            max_replays: 最大处理对局数（None表示处理全部）
        """
        if self.dataset is None:
            self.load_dataset()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_data = self.dataset['train']
        total_replays = len(train_data) if max_replays is None else min(max_replays, len(train_data))
        
        processed_count = 0
        
        for i in range(total_replays):
            try:
                replay_data = train_data[i]
                game_states = self.parse_replay(replay_data)
                
                # 保存处理后的数据
                np.savez_compressed(
                    output_path / f"replay_{i:06d}.npz",
                    game_states=game_states,
                    metadata={
                        'replay_id': replay_data.get('id', f'replay_{i}'),
                        'usernames': replay_data['usernames'],
                        'stars': replay_data['stars'],
                        'map_size': (replay_data['mapWidth'], replay_data['mapHeight'])
                    }
                )
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    self.logger.info(f"已处理 {processed_count}/{total_replays} 个对局")
                    
            except Exception as e:
                self.logger.error(f"处理对局 {i} 失败: {e}")
                continue
        
        self.logger.info(f"数据预处理完成，共处理 {processed_count} 个对局")
        return processed_count