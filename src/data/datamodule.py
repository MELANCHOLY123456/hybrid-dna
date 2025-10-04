import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os
from .dataset import TrueStreamingDataset


class PenetranceDataModule(pl.LightningDataModule):
    """Data module for gene variant penetrance prediction.
    
    管理训练、验证和测试数据的加载，采用流式处理策略以高效处理大规模数据集。
    """

    def __init__(self, cfg):
        """Initialize data module.
        
        Args:
            cfg: Configuration object 配置对象，包含数据集和加载器的配置参数
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.batch_files = None
        self.samples_per_file = None
        self.total_samples = 0
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._has_setup = False

    def prepare_data(self):
        """Prepare data stage: scan and validate data files.
        
        在单个进程中运行，扫描数据文件并验证完整性，不执行实际的数据加载。
        """
        model_name = "hyenadna-small-32k-seqlen"
        
        # Determine data directory path
        # 确定数据目录路径
        if hasattr(self.cfg.data, 'data_dir'):
            data_dir = self.cfg.data.data_dir
        else:
            # Use project root data directory as default
            # 使用项目根目录下的data目录作为默认路径
            import hydra
            data_dir = os.path.join(hydra.utils.get_original_cwd(), "data")
        
        # Build processed data directory path
        # 构建处理后的数据目录路径
        processed_dir = os.path.abspath(os.path.join(data_dir, "processed"))
        
        print("Scanning batch files...")
        print(f"Scanning directory: {processed_dir}")
        
        # Validate directory existence
        # 验证目录是否存在
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Directory not found: {processed_dir}")
        
        # Check directory access permissions
        # 检查目录访问权限
        try:
            all_files = os.listdir(processed_dir)
            print(f"Total files in directory: {len(all_files)}")
        except PermissionError:
            raise PermissionError(f"No permission to access directory: {processed_dir}")
        except Exception as e:
            raise Exception(f"Error accessing directory: {processed_dir}, error: {str(e)}")
        
        # Collect batch files matching pattern with sample counts
        # 收集匹配模式的批次文件及其样本数量
        batch_files_with_counts = []
        pattern = f"variants_with_embeddings_{model_name}_batch_"
        
        for filename in all_files:
            if filename.startswith(pattern) and filename.endswith('.npz'):
                batch_idx = int(filename.replace(pattern, '').replace('.npz', ''))
                file_path = os.path.join(processed_dir, filename)
                # Read actual sample count from file
                # 从文件中读取实际样本数量
                with np.load(file_path) as data:
                    sample_count = len(data['varids'])
                batch_files_with_counts.append((batch_idx, file_path, sample_count))
        
        # Sort by batch index
        # 按批次索引排序
        batch_files_with_counts.sort(key=lambda x: x[0])
        
        # Validate matching files found
        # 验证是否找到匹配的文件
        if not batch_files_with_counts:
            # List all files for debugging
            # 列出所有文件用于调试
            print(f"All files in directory {processed_dir}:")
            for file in all_files:
                print(f"  {file}")
            raise FileNotFoundError(f"No batch files found matching pattern: {pattern}*.npz")
        
        # Save file paths and actual sample counts
        # 保存文件路径和实际样本数量
        self.batch_files = [file_path for _, file_path, _ in batch_files_with_counts]
        self.samples_per_file = [count for _, _, count in batch_files_with_counts]
        self.total_samples = sum(self.samples_per_file)
        
        print(f"Scan completed:")
        print(f"   Batch files: {len(self.batch_files)}")
        print(f"   Total samples: {self.total_samples:,}")

    def setup(self, stage=None):
        """Setup datasets: create dataset objects and split train/val/test.
        
        Args:
            stage: Training stage ('fit', 'validate', 'test', 'predict')
        """
        # Prevent duplicate initialization across stages
        # 防止在不同阶段重复初始化
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Ensure prepare_data has been called
        # 确保prepare_data已被调用
        if self.batch_files is None or self.samples_per_file is None:
            self.prepare_data()

        if stage == "fit" or stage is None:
            # Create streaming dataset
            # 创建流式数据集
            self.full_dataset = TrueStreamingDataset(
                batch_files=self.batch_files,
                samples_per_file=self.samples_per_file,
                annotation_columns=self.cfg.data.annotation_columns,
                gene_columns=self.cfg.data.gene_columns,
                num_workers=self.cfg.loader.num_workers
            )

            # Split dataset according to configuration ratios
            # 根据配置比例分割数据集
            train_len = int(len(self.full_dataset) * self.cfg.data.train_val_test_split[0])
            val_len = int(len(self.full_dataset) * self.cfg.data.train_val_test_split[1])
            test_len = len(self.full_dataset) - train_len - val_len

            # Use sequential split instead of random split to maintain file order
            # 使用顺序分割而不是随机分割以保持文件顺序
            from torch.utils.data import Subset
            
            # Create sequential indices for train/val/test split
            # 为训练/验证/测试分割创建顺序索引
            train_indices = list(range(0, train_len))
            val_indices = list(range(train_len, train_len + val_len))
            test_indices = list(range(train_len + val_len, len(self.full_dataset)))
            
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
            self.test_dataset = Subset(self.full_dataset, test_indices)
            
            print(f"Dataset split:")
            print(f"   Train: {train_len:,} samples")
            print(f"   Validation: {val_len:,} samples") 
            print(f"   Test: {test_len:,} samples")

    def train_dataloader(self):
        """Create training data loader.
        
        Returns:
            DataLoader for training data 训练数据的DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.loader.batch_size,
            num_workers=self.cfg.loader.num_workers,
            shuffle=False,  # 关闭随机采样以配合单文件内存策略，减少文件切换
            pin_memory=self.cfg.loader.pin_memory,
            persistent_workers=True if self.cfg.loader.num_workers > 0 else False,
            drop_last=self.cfg.loader.drop_last
        )

    def val_dataloader(self):
        """Create validation data loader.
        
        Returns:
            DataLoader for validation data 验证数据的DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.loader.batch_size,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory,
            persistent_workers=True if self.cfg.loader.num_workers > 0 else False,
            drop_last=False
        )

    def test_dataloader(self):
        """Create test data loader.
        
        Returns:
            DataLoader for test data 测试数据的DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.loader.batch_size,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory,
            persistent_workers=True if self.cfg.loader.num_workers > 0 else False,
            drop_last=False
        )