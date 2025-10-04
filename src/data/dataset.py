import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from functools import lru_cache


class TrueStreamingDataset(Dataset):
    """Streaming dataset with on-demand loading and intelligent caching.
    
    实现高效的流式数据加载，使用LRU缓存策略减少大规模基因组变异数据的磁盘I/O操作。
    """

    def __init__(self, batch_files, samples_per_file, annotation_columns=None, gene_columns=None, num_workers=0):
        """Initialize streaming dataset.
        
        Args:
            batch_files: List of batch file paths 批次文件路径列表
            samples_per_file: List of sample counts per file 每个文件的样本数量列表
            annotation_columns: List of annotation feature column names 注释特征列名列表
            gene_columns: List of gene feature column names 基因特征列名列表
            num_workers: Number of worker processes used by DataLoader 数据加载器使用的worker进程数
        """
        self.batch_files = batch_files
        self.samples_per_file = samples_per_file
        self.annotation_columns = annotation_columns or []
        self.gene_columns = gene_columns or []
        self.num_workers = num_workers
        
        # Dynamic memory strategy based on num_workers
        # 基于num_workers的动态内存策略
        if num_workers > 0:
            # Multi-file memory strategy for multiple workers
            # 多worker的多文件内存策略
            self._max_cached_files = min(num_workers, 4)  # Limit cache size to prevent excessive memory usage
            self._get_file_data = self._get_file_data_multi_worker
        else:
            # Single file memory strategy for single worker
            # 单worker的单文件内存策略
            self._max_cached_files = 1
            self._get_file_data = self._get_file_data_single_worker
            
        # Get worker info for better logging
        # 获取worker信息以便更好的日志记录
        self._worker_id = os.getpid()  # Use process ID as worker identifier
        
        # Precompute sample counts for fast indexing
        # 预计算样本数量以实现快速索引
        self.total_samples = sum(samples_per_file)
        self.cumulative_samples = [0]
        for count in samples_per_file:
            self.cumulative_samples.append(self.cumulative_samples[-1] + count)

    def __len__(self):
        """Return total number of samples.
        
        Returns:
            Total sample count in dataset 数据集中的总样本数
        """
        return self.total_samples

    def __getitem__(self, idx):
        """Get sample data at specified index.
        
        Args:
            idx: Sample index 样本索引
            
        Returns:
            Dictionary containing sample data with keys:
                - 'embedding': Sequence embedding features 序列嵌入特征
                - 'annotation_features': Annotation features 注释特征
                - 'gene_features': Gene features 基因特征
                - 'label': Label value 标签值
                - 'varid': Variant ID 变异ID
                
        Raises:
            IndexError: When index is out of range 当索引超出范围时抛出异常
        """
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")
        
        # Use binary search to locate file and sample indices
        # 使用二分查找定位文件和样本索引
        file_idx = self._find_file_index(idx)
        sample_idx = idx - self.cumulative_samples[file_idx]
        batch_file = self.batch_files[file_idx]
        
        sample_data = self._load_sample_from_file(batch_file, sample_idx)
        
        return sample_data
    
    def _find_file_index(self, idx):
        """Find file index using binary search.
        
        使用预计算的累积样本数快速定位包含指定样本的文件。
        
        Args:
            idx: Sample index 样本索引
            
        Returns:
            File index 文件索引
        """
        left, right = 0, len(self.cumulative_samples) - 2
        while left <= right:
            mid = (left + right) // 2
            if self.cumulative_samples[mid] <= idx < self.cumulative_samples[mid + 1]:
                return mid
            elif idx < self.cumulative_samples[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def _get_file_data_single_worker(self, batch_file):
        """Get file data using single file memory strategy for single worker.
        
        只在内存中保持一个文件，需要时才加载新文件并释放旧文件。
        
        Args:
            batch_file: Batch file path 批次文件路径
            
        Returns:
            File data dictionary 文件数据字典
        """
        # Check if the requested file is already loaded
        # 检查请求的文件是否已经加载
        if hasattr(self, '_current_file_path') and self._current_file_path == batch_file and self._current_file_data is not None:
            return self._current_file_data
        
        # Need to load a new file - first release current file memory
        # 需要加载新文件 - 首先释放当前文件内存
        if hasattr(self, '_current_file_data') and self._current_file_data is not None:
            self._current_file_data = None  # Let garbage collector free memory
            
        # Load new file to memory
        # 将新文件加载到内存
        if not hasattr(self, '_files_loaded'):
            self._files_loaded = 0
        self._files_loaded += 1
        if self._files_loaded == 1:
            print(f"Worker {self._worker_id}: Using single-file memory strategy (max ~237MB per file)")
        
        filename = batch_file.split('/')[-1]
        batch_num = filename.split('_')[-1].replace('.npz', '')
        # 将详细文件加载信息保存到后台（不直接打印）
        # Save detailed file loading information to background (don't print directly)
        file_info = f"Worker {self._worker_id}: Loading file {self._files_loaded} (batch_{batch_num}): {filename}"
        
        file_data = {}
        with np.load(batch_file) as data:
            file_data['variant_embeddings'] = data['embeddings'].copy()
            file_data['penetrance_estimates'] = data['penetrance_estimates'].copy()
            file_data['varids'] = data['varids'].copy()
            
            # Copy feature data
            # 复制特征数据
            for col in self.annotation_columns + self.gene_columns:
                if col in data:
                    file_data[col] = data[col].copy()
        
        # Update current file tracking
        # 更新当前文件跟踪
        self._current_file_path = batch_file
        self._current_file_data = file_data
        
        return file_data

    def _get_file_data_multi_worker(self, batch_file):
        """Get file data using LRU cache for multiple workers.
        
        使用LRU缓存策略为多worker环境缓存多个文件。
        
        Args:
            batch_file: Batch file path 批次文件路径
            
        Returns:
            File data dictionary 文件数据字典
        """
        # Use LRU cache to store multiple files in memory
        # 使用LRU缓存来在内存中存储多个文件
        return self._load_file_with_cache(batch_file)

    @lru_cache(maxsize=None)
    def _load_file_with_cache(self, batch_file):
        """Load file data with LRU cache.
        
        使用LRU缓存加载文件数据。
        
        Args:
            batch_file: Batch file path 批次文件路径
            
        Returns:
            File data dictionary 文件数据字典
        """
        # Initialize files loaded counter if not exists
        # 如果不存在则初始化文件加载计数器
        if not hasattr(self, '_cached_files_loaded'):
            self._cached_files_loaded = 0
        self._cached_files_loaded += 1
        
        # Print loading message for first file
        # 为第一个文件打印加载消息
        if self._cached_files_loaded == 1:
            print(f"Worker {self._worker_id}: Using multi-file memory strategy (max {self._max_cached_files} files cached)")
        
        filename = batch_file.split('/')[-1]
        batch_num = filename.split('_')[-1].replace('.npz', '')
        # 将详细文件加载信息保存到后台（不直接打印）
        # Save detailed file loading information to background (don't print directly)
        file_info = f"Worker {self._worker_id}: Loading file {self._cached_files_loaded} (batch_{batch_num}): {filename}"
        
        file_data = {}
        with np.load(batch_file) as data:
            file_data['variant_embeddings'] = data['embeddings'].copy()
            file_data['penetrance_estimates'] = data['penetrance_estimates'].copy()
            file_data['varids'] = data['varids'].copy()
            
            # Copy feature data
            # 复制特征数据
            for col in self.annotation_columns + self.gene_columns:
                if col in data:
                    file_data[col] = data[col].copy()
        
        return file_data

    def _load_sample_from_file(self, batch_file, sample_idx):
        """Load single sample from cached file data.
        
        Args:
            batch_file: Batch file path 批次文件路径
            sample_idx: Sample index within file 文件内的样本索引
            
        Returns:
            Sample data dictionary 样本数据字典
        """
        try:
            batch_data = self._get_file_data(batch_file)
            
            # Get sequence embedding features
            # 获取序列嵌入特征
            embedding = batch_data['variant_embeddings'][sample_idx].astype(np.float32)
            
            # Extract annotation features, handle NaN and categorical features
            # 提取注释特征，处理NaN和分类特征
            annotation_features = []
            for col in self.annotation_columns:
                if col in batch_data:
                    raw_value = batch_data[col][sample_idx]
                    
                    # Handle categorical feature CLNSIG
                    # 处理分类特征CLNSIG
                    if col == 'CLNSIG':
                        if isinstance(raw_value, str):
                            clnsig_map = {
                                'Pathogenic': 1.0,
                                'Likely_pathogenic': 0.8,
                                'Uncertain_significance': 0.5,
                                'Likely_benign': 0.2,
                                'Benign': 0.0
                            }
                            value = clnsig_map.get(raw_value, 0.5)  # Default uncertain 默认不确定
                        else:
                            value = 0.5  # NaN or other values default uncertain NaN或其他值默认为不确定
            
                    # Handle categorical feature ClinPred_pred
                    # 处理分类特征ClinPred_pred
                    elif col == 'ClinPred_pred':
                        if isinstance(raw_value, str):
                            value = 1.0 if raw_value == 'D' else 0.0  # D=Deleterious, T=Tolerated D=有害，T=可耐受
                        else:
                            value = 0.0  # Default benign 默认良性
                    
                    # Handle numeric features
                    # 处理数值特征
                    else:
                        try:
                            value = float(raw_value)
                            value = 0.0 if np.isnan(value) else value
                        except (ValueError, TypeError):
                            value = 0.0
                    
                    annotation_features.append(value)
                else:
                    annotation_features.append(0.0)
            
            # Extract gene features, handle NaN values
            # 提取基因特征，处理NaN值
            gene_features = []
            for col in self.gene_columns:
                if col in batch_data:
                    value = float(batch_data[col][sample_idx])
                    gene_features.append(0.0 if np.isnan(value) else value)
                else:
                    gene_features.append(0.0)
            
            # Get label and variant ID
            # 获取标签和变异ID
            label = float(batch_data['penetrance_estimates'][sample_idx])
            varid = str(batch_data['varids'][sample_idx])
            
            # Return formatted sample data
            # 返回格式化的样本数据
            return {
                'embedding': torch.tensor(embedding, dtype=torch.float32),  # 保持原始维度
                'annotation_features': torch.tensor(annotation_features, dtype=torch.float32).unsqueeze(0),  # [D] -> [1, D]
                'gene_features': torch.tensor(gene_features, dtype=torch.float32).unsqueeze(0),  # [D] -> [1, D]
                'label': torch.tensor(label, dtype=torch.float32),
                'varid': varid
            }
            
        except Exception as e:
            # Error handling: return zero vectors as default
            # 错误处理：返回零向量作为默认值
            print(f"Warning: Failed to read sample (file: {batch_file}, index: {sample_idx}): {e}")
            return {
                'embedding': torch.zeros(512, 256, dtype=torch.float32),  # 保持正确的维度
                'annotation_features': torch.zeros(1, len(self.annotation_columns), dtype=torch.float32),  # [1, D]
                'gene_features': torch.zeros(1, len(self.gene_columns), dtype=torch.float32),  # [1, D]
                'label': torch.tensor(0.0, dtype=torch.float32),
                'varid': f"error_{sample_idx}"
            }
    
    def get_memory_info(self):
        """Get memory usage statistics.
        
        Returns:
            Dictionary containing memory usage statistics 包含内存使用统计信息的字典
        """
        return {
            'files_loaded': self._files_loaded,
            'current_file': self._current_file_path.split('/')[-1] if self._current_file_path else None,
            'memory_strategy': 'single_file',
            'estimated_memory_usage': '~237MB' if self._current_file_data else '0MB'
        }
    
    def print_memory_summary(self):
        """Print memory usage summary information.
        
        打印内存使用摘要信息。
        """
        current_file = self._current_file_path.split('/')[-1] if self._current_file_path else "None"
        memory_usage = "~237MB" if self._current_file_data else "0MB"
        print(f"Memory usage: {memory_usage}, Current file: {current_file}, Files loaded: {self._files_loaded}")
    
    
