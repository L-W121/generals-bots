"""
快速修复脚本 - 立即解决 PyTorch DataLoader 尺寸问题
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

class QuickFixDataset(Dataset):
    """快速修复版本 - 只使用最常见的地图尺寸"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        # 目标尺寸：最常见的 21×18 (height × width)
        target_h, target_w = 21, 18
        
        print("🔧 快速修复：过滤相同尺寸样本...")
        
        # 找到批次文件
        batch_files = list(self.data_dir.glob("batch_*.pt"))
        if not batch_files:
            batch_files = list(self.data_dir.glob("optimized_batch_*.pt"))
        
        total_samples = 0
        valid_samples = 0
        
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file, map_location='cpu')
                
                for state_tensor, action_id, metadata in batch_data:
                    total_samples += 1
                    _, h, w = state_tensor.shape
                    
                    # 只保留 21×18 尺寸的样本
                    if h == target_h and w == target_w:
                        self.samples.append((state_tensor, action_id))
                        valid_samples += 1
                
            except Exception as e:
                print(f"⚠️  跳过文件: {batch_file.name}")
                continue
        
        print(f"✅ 修复完成!")
        print(f"   原始样本: {total_samples:,}")
        print(f"   有效样本: {valid_samples:,} (21×18 尺寸)")
        print(f"   保留比例: {valid_samples/total_samples*100:.1f}%")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def quick_test(data_dir: str):
    """快速测试修复效果"""
    print("🧪 快速测试修复效果...")
    
    try:
        # 创建修复后的数据集
        dataset = QuickFixDataset(data_dir)
        
        if len(dataset) == 0:
            print("❌ 没有找到21×18尺寸的样本，尝试其他尺寸...")
            return False
        
        # 测试 DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 获取一个批次
        batch_states, batch_actions = next(iter(dataloader))
        
        print(f"✅ 测试成功!")
        print(f"   批次状态形状: {batch_states.shape}")
        print(f"   批次动作形状: {batch_actions.shape}")
        print(f"   数据类型: {batch_states.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def analyze_map_sizes(data_dir: str):
    """分析地图尺寸分布"""
    print("📊 分析地图尺寸分布...")
    
    data_path = Path(data_dir)
    batch_files = list(data_path.glob("batch_*.pt")) + list(data_path.glob("optimized_batch_*.pt"))
    
    size_count = {}
    total_samples = 0
    
    for batch_file in batch_files[:3]:  # 只分析前3个文件
        try:
            batch_data = torch.load(batch_file, map_location='cpu')
            
            for state_tensor, action_id, metadata in batch_data:
                total_samples += 1
                _, h, w = state_tensor.shape
                size = (h, w)
                size_count[size] = size_count.get(size, 0) + 1
                
        except Exception:
            continue
    
    print(f"\\n📏 地图尺寸分布 (基于 {total_samples} 个样本):")
    for size, count in sorted(size_count.items(), key=lambda x: -x[1]):
        h, w = size
        percentage = count / total_samples * 100
        print(f"   {h}×{w}: {count:,} 样本 ({percentage:.1f}%)")
    
    # 推荐最佳尺寸
    if size_count:
        best_size = max(size_count.items(), key=lambda x: x[1])
        print(f"\\n🎯 推荐使用: {best_size[0][0]}×{best_size[0][1]} (最多样本)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速修复 DataLoader 尺寸问题')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径')
    parser.add_argument('--analyze', action='store_true',
                       help='分析地图尺寸分布')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_map_sizes(args.data_dir)
    else:
        success = quick_test(args.data_dir)
        if not success:
            print("\\n🔍 建议运行分析:")
            print(f"python {__file__} --data_dir {args.data_dir} --analyze")