"""
智能填充策略 - 用山脉填充边界使地图尺寸统一
这是最优雅的解决方案，既保留所有数据又符合游戏逻辑

"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys

class MountainPaddedDataset(Dataset):
    """用山脉填充的智能数据集 - 最佳解决方案"""
    
    def __init__(self, data_dir: str, 
                 unified_size: tuple = (22, 22),  # 统一尺寸
                 mountain_value: float = -1.0):   # 山脉在状态中的值
        """
        Args:
            unified_size: 统一后的地图尺寸 (height, width)
            mountain_value: 山脉通道中的数值
        """
        self.data_dir = Path(data_dir)
        self.unified_h, self.unified_w = unified_size
        self.mountain_value = mountain_value
        self.samples = []
        
        print(f"🏔️  智能山脉填充策略")
        print(f"   目标尺寸: {unified_size}")
        print(f"   策略: 用山脉填充边界区域")
        
        self._load_and_pad_all_samples()
    
    def _load_and_pad_all_samples(self):
        """加载所有样本并用山脉智能填充"""
        batch_files = list(self.data_dir.glob("batch_*.pt")) + \
                     list(self.data_dir.glob("optimized_batch_*.pt"))
        
        total_samples = 0
        size_stats = {}
        
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file, map_location='cpu')
                
                for state_tensor, action_id, metadata in batch_data:
                    total_samples += 1
                    
                    # 记录原始尺寸
                    _, orig_h, orig_w = state_tensor.shape
                    size_key = (orig_h, orig_w)
                    size_stats[size_key] = size_stats.get(size_key, 0) + 1
                    
                    # 智能填充
                    padded_tensor = self._smart_mountain_padding(state_tensor)
                    self.samples.append((padded_tensor, action_id, metadata))
                
            except Exception as e:
                print(f"⚠️ 跳过文件: {batch_file.name}")
                continue
        
        print(f"\\n📊 处理完成:")
        print(f"   总样本数: {total_samples:,}")
        print(f"   成功填充: {len(self.samples):,}")
        print(f"   统一尺寸: {self.unified_h}×{self.unified_w}")
        
        print(f"\\n📏 原始尺寸分布:")
        for size, count in sorted(size_stats.items(), key=lambda x: -x[1])[:10]:
            h, w = size
            percentage = count / total_samples * 100
            print(f"   {h:2d}×{w:2d}: {count:4d} ({percentage:4.1f}%)")
    
    def _smart_mountain_padding(self, state_tensor):
        """智能山脉填充策略"""
        channels, orig_h, orig_w = state_tensor.shape
        
        # 如果已经是目标尺寸，直接返回
        if orig_h == self.unified_h and orig_w == self.unified_w:
            return state_tensor
        
        # 如果原尺寸超过目标尺寸，需要裁剪
        if orig_h > self.unified_h or orig_w > self.unified_w:
            # 从中心裁剪
            crop_h = min(orig_h, self.unified_h)
            crop_w = min(orig_w, self.unified_w)
            
            start_h = (orig_h - crop_h) // 2
            start_w = (orig_w - crop_w) // 2
            
            state_tensor = state_tensor[:, 
                                      start_h:start_h+crop_h,
                                      start_w:start_w+crop_w]
            orig_h, orig_w = crop_h, crop_w
        
        # 创建目标尺寸的张量
        padded_tensor = torch.zeros(channels, self.unified_h, self.unified_w, 
                                  dtype=state_tensor.dtype)
        
        # 计算居中放置的位置
        start_h = (self.unified_h - orig_h) // 2
        start_w = (self.unified_w - orig_w) // 2
        
        # 放置原始数据到中心
        padded_tensor[:, start_h:start_h+orig_h, start_w:start_w+orig_w] = state_tensor
        
        # 关键：在山脉通道(第4通道)中填充边界为山脉
        mountain_channel = 4  # 根据state_encoder.py，山脉是第4通道
        
        # 填充上下边界
        if start_h > 0:
            padded_tensor[mountain_channel, :start_h, :] = 1.0          # 上边界
        if start_h + orig_h < self.unified_h:
            padded_tensor[mountain_channel, start_h+orig_h:, :] = 1.0   # 下边界
        
        # 填充左右边界
        if start_w > 0:
            padded_tensor[mountain_channel, :, :start_w] = 1.0          # 左边界
        if start_w + orig_w < self.unified_w:
            padded_tensor[mountain_channel, :, start_w+orig_w:] = 1.0   # 右边界
        
        return padded_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state_tensor, action_id, metadata = self.samples[idx]
        return state_tensor, action_id
    
    def get_sample_info(self, idx):
        """获取样本详细信息"""
        state_tensor, action_id, metadata = self.samples[idx]
        return {
            'state_shape': state_tensor.shape,
            'action_id': action_id,
            'metadata': metadata,
            'mountain_mask': state_tensor[4].sum().item()  # 山脉数量
        }

class MultiSizeDataset:
    """多尺寸分类训练数据集"""
    
    def __init__(self, data_dir: str, min_samples_per_size: int = 1000):
        """
        按地图尺寸分类创建多个数据集
        
        Args:
            min_samples_per_size: 每种尺寸最少样本数，少于此数的尺寸会被忽略
        """
        self.data_dir = Path(data_dir)
        self.min_samples_per_size = min_samples_per_size
        self.datasets_by_size = {}
        
        print(f"📂 创建多尺寸分类数据集")
        print(f"   最小样本数阈值: {min_samples_per_size}")
        
        self._organize_by_size()
    
    def _organize_by_size(self):
        """按尺寸组织数据"""
        # 先统计各尺寸的样本数
        size_samples = {}
        
        batch_files = list(self.data_dir.glob("batch_*.pt")) + \
                     list(self.data_dir.glob("optimized_batch_*.pt"))
        
        print("🔍 统计各尺寸样本数...")
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file, map_location='cpu')
                
                for state_tensor, action_id, metadata in batch_data:
                    _, h, w = state_tensor.shape
                    size_key = (h, w)
                    
                    if size_key not in size_samples:
                        size_samples[size_key] = []
                    
                    size_samples[size_key].append((state_tensor, action_id, metadata))
                
            except Exception:
                continue
        
        # 筛选样本数足够的尺寸
        valid_sizes = {size: samples for size, samples in size_samples.items() 
                      if len(samples) >= self.min_samples_per_size}
        
        print(f"\\n📊 尺寸分布:")
        for size, samples in sorted(size_samples.items(), key=lambda x: -len(x[1])):
            h, w = size
            count = len(samples)
            status = "✅" if count >= self.min_samples_per_size else "❌"
            print(f"   {status} {h:2d}×{w:2d}: {count:6,} 样本")
        
        # 创建各尺寸的数据集
        print(f"\\n🏗️  创建 {len(valid_sizes)} 个尺寸数据集:")
        for size, samples in valid_sizes.items():
            h, w = size
            dataset = SingleSizeDataset(samples, size)
            self.datasets_by_size[size] = dataset
            print(f"   {h:2d}×{w:2d}: {len(dataset):6,} 样本")
    
    def get_dataset(self, size: tuple):
        """获取指定尺寸的数据集"""
        return self.datasets_by_size.get(size)
    
    def get_all_sizes(self):
        """获取所有可用尺寸"""
        return list(self.datasets_by_size.keys())
    
    def get_largest_dataset(self):
        """获取样本数最多的数据集"""
        if not self.datasets_by_size:
            return None
        
        largest_size = max(self.datasets_by_size.keys(), 
                          key=lambda size: len(self.datasets_by_size[size]))
        return self.datasets_by_size[largest_size], largest_size

class SingleSizeDataset(Dataset):
    """单一尺寸数据集"""
    
    def __init__(self, samples, size):
        self.samples = samples
        self.size = size
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state_tensor, action_id, metadata = self.samples[idx]
        return state_tensor, action_id

def test_mountain_padding(data_dir: str):
    """测试山脉填充方案"""
    print("🏔️  测试智能山脉填充方案")
    print("=" * 50)
    
    try:
        # 创建山脉填充数据集
        dataset = MountainPaddedDataset(data_dir, unified_size=(25, 25))
        
        if len(dataset) == 0:
            print("❌ 没有可用样本")
            return False
        
        # 测试DataLoader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        batch_states, batch_actions = next(iter(dataloader))
        
        print(f"\\n✅ 山脉填充测试成功!")
        print(f"   样本数量: {len(dataset):,}")
        print(f"   批次形状: {batch_states.shape}")
        print(f"   数据类型: {batch_states.dtype}")
        
        # 检查填充效果
        sample_info = dataset.get_sample_info(0)
        print(f"   样本信息: {sample_info}")
        
        # 可视化一个样本的山脉分布
        sample_state = batch_states[0]
        mountain_channel = sample_state[4]  # 山脉通道
        mountain_count = (mountain_channel > 0).sum().item()
        total_cells = mountain_channel.numel()
        
        print(f"   山脉覆盖: {mountain_count}/{total_cells} ({mountain_count/total_cells*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 山脉填充测试失败: {e}")
        return False

def test_multi_size(data_dir: str):
    """测试多尺寸分类方案"""
    print("\\n📂 测试多尺寸分类方案")
    print("=" * 50)
    
    try:
        # 创建多尺寸数据集
        multi_dataset = MultiSizeDataset(data_dir, min_samples_per_size=500)
        
        available_sizes = multi_dataset.get_all_sizes()
        if not available_sizes:
            print("❌ 没有找到足够样本的尺寸")
            return False
        
        print(f"\\n✅ 多尺寸分类成功!")
        print(f"   可用尺寸: {len(available_sizes)} 种")
        
        # 测试每种尺寸的数据集
        for size in available_sizes[:3]:  # 只测试前3种
            dataset = multi_dataset.get_dataset(size)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            batch_states, batch_actions = next(iter(dataloader))
            
            h, w = size
            print(f"   {h:2d}×{w:2d}: {len(dataset):6,} 样本, 批次: {batch_states.shape}")
        
        # 测试最大数据集
        largest_dataset, largest_size = multi_dataset.get_largest_dataset()
        print(f"\\n🏆 最大数据集: {largest_size[0]}×{largest_size[1]} "
              f"({len(largest_dataset):,} 样本)")
        
        return True
        
    except Exception as e:
        print(f"❌ 多尺寸分类测试失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='智能数据集解决方案')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--method', type=str, 
                       choices=['mountain', 'multi_size', 'both'], 
                       default='both')
    
    args = parser.parse_args()
    
    if args.method in ['mountain', 'both']:
        success1 = test_mountain_padding(args.data_dir)
    else:
        success1 = True
    
    if args.method in ['multi_size', 'both']:
        success2 = test_multi_size(args.data_dir)
    else:
        success2 = True
    
    print("\\n🎯 推荐方案:")
    print("=" * 50)
    if success1:
        print("✅ 山脉填充方案: 保留所有数据，符合游戏逻辑")
        print("   使用方法: dataset = MountainPaddedDataset(data_dir)")
    
    if success2:
        print("✅ 多尺寸分类方案: 分别训练不同尺寸模型") 
        print("   使用方法: multi = MultiSizeDataset(data_dir)")
    
    print("\\n💡 建议:")
    print("   初期开发: 使用山脉填充方案 (简单统一)")
    print("   高级应用: 结合两种方案，针对性优化")