"""
数据预处理主脚本
整合所有预处理组件，生成训练样本
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入自定义模块（这些文件应该放在 data/preprocessing/ 目录下）
from data_process.preprocessing.replay_parser import GeneralsReplayParser
from data_process.preprocessing.state_encoder import StateEncoder  
from data_process.preprocessing.action_encoder import ActionEncoder

class GeneralsTrainingDataset:
    """Generals.io 训练数据集生成器"""
    
    def __init__(self, 
                 dataset_name: str = "strakammm/generals_io_replays",
                 cache_dir: Optional[str] = None,
                 max_replays: Optional[int] = None):
        """
        初始化数据集生成器
        
        Args:
            dataset_name: Hugging Face 数据集名称
            cache_dir: 缓存目录
            max_replays: 最大处理对局数
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.max_replays = max_replays
        
        # 初始化组件
        self.replay_parser = GeneralsReplayParser(dataset_name, cache_dir)
        self.state_encoder = StateEncoder(num_channels=15)
        
        # 动作编码器将根据地图尺寸动态创建
        self.action_encoders = {}
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_preprocessing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def get_or_create_action_encoder(self, width: int, height: int) -> ActionEncoder:
        """获取或创建指定尺寸的动作编码器"""
        key = (width, height)
        if key not in self.action_encoders:
            self.action_encoders[key] = ActionEncoder(width, height)
        return self.action_encoders[key]
    
    def process_single_replay(self, replay_data: Dict) -> List[Tuple]:
        """
        处理单个对局，生成训练样本
        
        Args:
            replay_data: 对局数据
            
        Returns:
            训练样本列表 [(state_tensor, action_id, metadata), ...]
        """
        try:
            width = replay_data['mapWidth']
            height = replay_data['mapHeight'] 
            moves = replay_data['moves']
            
            # 获取动作编码器
            action_encoder = self.get_or_create_action_encoder(width, height)
            
            # 解析对局状态序列
            game_states = self.replay_parser.parse_replay(replay_data)
            
            if len(game_states) != len(moves) + 1:
                self.logger.warning(f"状态数量与移动数量不匹配")
                return []
            
            samples = []
            
            # 生成训练样本
            for i, move in enumerate(moves):
                if i >= len(game_states) - 1:
                    break
                
                player_id = move[0]
                current_state = game_states[i]
                
                # 编码状态
                state_tensor = self.state_encoder.encode_state(
                    current_state, player_id, normalize=True
                )
                # 在生成 state_tensor 后
                state_tensor = self._mountain_pad(state_tensor, target_h=22, target_w=22)

                
                # 编码动作
                try:
                    action_id = action_encoder.encode_move(move)
                except Exception as e:
                    self.logger.debug(f"动作编码失败: {move}, 错误: {e}")
                    continue
                
                # 创建元数据
                metadata = {
                    'replay_id': replay_data.get('id', 'unknown'),
                    'turn': move[4],
                    'player_id': player_id,
                    'map_size': (width, height),
                    'move': move
                }
                
                samples.append((state_tensor, action_id, metadata))
            
            return samples
            
        except Exception as e:
            self.logger.error(f"处理对局失败: {e}")
            return []
    
    def generate_training_data(self, output_dir: str) -> Dict:
        """
        生成训练数据
        
        Args:
            output_dir: 输出目录
            
        Returns:
            处理统计信息
        """
        self.logger.info("开始生成训练数据...")
        
        # 加载数据集
        self.replay_parser.load_dataset()
        dataset = self.replay_parser.dataset['train']
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理统计
        stats = {
            'total_replays_processed': 0,
            'total_samples_generated': 0,
            'map_size_distribution': {},
            'player_distribution': {},
            'failed_replays': 0,
            'avg_samples_per_replay': 0
        }
        
        # 确定处理数量
        total_replays = len(dataset)
        if self.max_replays:
            total_replays = min(total_replays, self.max_replays)
        
        self.logger.info(f"准备处理 {total_replays} 个对局")
        
        # 分批处理数据
        batch_size = 100
        all_samples = []
        
        for batch_start in tqdm(range(0, total_replays, batch_size), 
                              desc="处理对局批次"):
            batch_samples = []
            batch_end = min(batch_start + batch_size, total_replays)
            
            for i in range(batch_start, batch_end):
                try:
                    replay_data = dataset[i]
                    samples = self.process_single_replay(replay_data)
                    
                    if samples:
                        batch_samples.extend(samples)
                        stats['total_replays_processed'] += 1
                        
                        # 更新统计信息
                        map_size = (replay_data['mapWidth'], replay_data['mapHeight'])
                        stats['map_size_distribution'][str(map_size)] = \
                            stats['map_size_distribution'].get(str(map_size), 0) + 1
                        
                        for username in replay_data['usernames']:
                            stats['player_distribution'][username] = \
                                stats['player_distribution'].get(username, 0) + 1
                    else:
                        stats['failed_replays'] += 1
                        
                except Exception as e:
                    self.logger.error(f"处理对局 {i} 失败: {e}")
                    stats['failed_replays'] += 1
            
            # 保存批次数据
            if batch_samples:
                batch_file = output_path / f"batch_{batch_start:06d}_{batch_end:06d}.pt"
                torch.save(batch_samples, batch_file)
                all_samples.extend(batch_samples)
                
                self.logger.info(f"批次 {batch_start}-{batch_end} 完成，"
                               f"生成 {len(batch_samples)} 个样本")
        
        # 更新最终统计
        stats['total_samples_generated'] = len(all_samples)
        if stats['total_replays_processed'] > 0:
            stats['avg_samples_per_replay'] = \
                stats['total_samples_generated'] / stats['total_replays_processed']
        
        # 保存统计信息
        stats_file = output_path / "preprocessing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 转换 set 为 list 以便 JSON 序列化
            serializable_stats = self._make_serializable(stats)
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        # 保存样本索引（用于快速加载）
        self._create_sample_index(all_samples, output_path)
        
        self.logger.info(f"数据预处理完成！")
        self.logger.info(f"  - 处理对局数: {stats['total_replays_processed']}")
        self.logger.info(f"  - 生成样本数: {stats['total_samples_generated']}")
        self.logger.info(f"  - 平均每局样本数: {stats['avg_samples_per_replay']:.1f}")
        self.logger.info(f"  - 失败对局数: {stats['failed_replays']}")
        
        return stats
    
    def _make_serializable(self, obj):
        """使对象可序列化为JSON"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    def _create_sample_index(self, samples: List[Tuple], output_path: Path):
        """创建样本索引以便快速访问"""
        index_data = {
            'total_samples': len(samples),
            'samples_by_map_size': {},
            'samples_by_player': {},
            'sample_metadata': []
        }
        
        for i, (state_tensor, action_id, metadata) in enumerate(samples):
            # 按地图尺寸分组
            map_size = str(metadata['map_size'])
            if map_size not in index_data['samples_by_map_size']:
                index_data['samples_by_map_size'][map_size] = []
            index_data['samples_by_map_size'][map_size].append(i)
            
            # 按玩家分组  
            player_id = metadata['player_id']
            if player_id not in index_data['samples_by_player']:
                index_data['samples_by_player'][player_id] = []
            index_data['samples_by_player'][player_id].append(i)
            
            # 保存元数据
            index_data['sample_metadata'].append({
                'replay_id': metadata['replay_id'],
                'turn': metadata['turn'],
                'player_id': metadata['player_id'],
                'map_size': metadata['map_size']
            })
        
        # 保存索引
        index_file = output_path / "sample_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        
        self.logger.info(f"样本索引已保存到 {index_file}")
    
    def _mountain_pad(self, state_tensor: torch.Tensor, target_h: int = 22, target_w: int = 22) -> torch.Tensor:
        """
        用山脉（第4通道）填充边界，将 state_tensor 填充到 [C, target_h, target_w]
        """
        c, h, w = state_tensor.shape
        # 裁剪超出部分
        if h > target_h or w > target_w:
            h2 = min(h, target_h)
            w2 = min(w, target_w)
            state_tensor = state_tensor[:, :h2, :w2]
            h, w = h2, w2

        # 创建目标张量
        padded = torch.zeros(c, target_h, target_w, dtype=state_tensor.dtype)
        # 居中放置原始
        offset_h = (target_h - h) // 2
        offset_w = (target_w - w) // 2
        padded[:, offset_h:offset_h+h, offset_w:offset_w+w] = state_tensor

        # 山脉通道索引（第 4 通道）
        M = 4
        # 上边
        if offset_h > 0:
            padded[M, :offset_h, :] = 1.0
        # 下边
        if offset_h + h < target_h:
            padded[M, offset_h+h:, :] = 1.0
        # 左边
        if offset_w > 0:
            padded[M, :, :offset_w] = 1.0
        # 右边
        if offset_w + w < target_w:
            padded[M, :, offset_w+w:] = 1.0

        return padded

class GeneralsPyTorchDataset(Dataset):
    """PyTorch Dataset 包装器"""
    
    def __init__(self, data_dir: str, map_size_filter: Optional[Tuple[int, int]] = None):
        """
        初始化 PyTorch Dataset
        
        Args:
            data_dir: 预处理数据目录
            map_size_filter: 地图尺寸过滤器
        """
        self.data_dir = Path(data_dir)
        self.map_size_filter = map_size_filter
        
        # 加载样本索引
        index_file = self.data_dir / "sample_index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"未找到样本索引文件: {index_file}")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            self.index_data = json.load(f)
        
        # 应用地图尺寸过滤
        if map_size_filter:
            self.sample_indices = self.index_data['samples_by_map_size'].get(
                str(map_size_filter), []
            )
        else:
            self.sample_indices = list(range(self.index_data['total_samples']))
        
        # 加载所有批次文件
        self.batch_files = sorted(self.data_dir.glob("batch_*.pt"))
        self.samples = []
        
        for batch_file in self.batch_files:
            batch_samples = torch.load(batch_file, map_location='cpu')
            self.samples.extend(batch_samples)
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx = self.sample_indices[idx]
        state_tensor, action_id, metadata = self.samples[sample_idx]
        return state_tensor, action_id

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generals.io 数据预处理')
    parser.add_argument('--output_dir', type=str, default='data/cache/preprocessed',
                       help='输出目录')
    parser.add_argument('--max_replays', type=int, default=None,
                       help='最大处理对局数')
    parser.add_argument('--dataset_name', type=str, 
                       default='strakammm/generals_io_replays',
                       help='Hugging Face 数据集名称')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='数据集缓存目录')
    
    args = parser.parse_args()
    
    # 创建数据集生成器
    dataset_generator = GeneralsTrainingDataset(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        max_replays=args.max_replays
    )
    
    # 生成训练数据
    stats = dataset_generator.generate_training_data(args.output_dir)
    
    # 测试 PyTorch Dataset
    print("\\n测试 PyTorch Dataset 加载...")
    try:
        pytorch_dataset = GeneralsPyTorchDataset(args.output_dir)
        print(f"成功创建 PyTorch Dataset，包含 {len(pytorch_dataset)} 个样本")
        
        # 测试数据加载
        if len(pytorch_dataset) > 0:
            state, action = pytorch_dataset[0]
            print(f"样本形状: 状态={state.shape}, 动作={action}")
            
            # 创建 DataLoader 测试
            dataloader = DataLoader(pytorch_dataset, batch_size=32, shuffle=True)
            batch_states, batch_actions = next(iter(dataloader))
            print(f"批次形状: 状态={batch_states.shape}, 动作={batch_actions.shape}")
        
    except Exception as e:
        print(f"PyTorch Dataset 测试失败: {e}")

if __name__ == "__main__":
    main()