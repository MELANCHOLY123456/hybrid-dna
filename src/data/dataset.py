import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TrueStreamingDataset(Dataset):
    """Streaming dataset with on-demand loading and intelligent caching.
    
    实现高效的流式数据加载，使用LRU缓存策略减少大规模基因组变异数据的磁盘I/O操作。
    """

    def __init__(self, batch_files, samples_per_file, annotation_columns=None, gene_columns=None):
        """Initialize streaming dataset.
        
        Args:
            batch_files: List of batch file paths 批次文件路径列表
            samples_per_file: List of sample counts per file 每个文件的样本数量列表
            annotation_columns: List of annotation feature column names 注释特征列名列表
            gene_columns: List of gene feature column names 基因特征列名列表
        """
        self.batch_files = batch_files
        self.samples_per_file = samples_per_file
        self.annotation_columns = annotation_columns or []
        self.gene_columns = gene_columns or []
        
        # LRU file cache for recently accessed files
        # 最近访问文件的LRU缓存
        from collections import OrderedDict
        self._file_cache = OrderedDict()
        self._max_cache_size = 50
        self._cache_hits = 0
        self._cache_misses = 0
        
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

    def _get_file_data(self, batch_file):
        """Get file data using LRU cache strategy.
        
        先检查缓存，如果存在则返回；否则加载文件数据并更新缓存。
        
        Args:
            batch_file: Batch file path 批次文件路径
            
        Returns:
            File data dictionary 文件数据字典
        """
        # Check if file data exists in cache
        # 检查文件数据是否存在于缓存中
        if batch_file in self._file_cache:
            self._cache_hits += 1
            self._file_cache.move_to_end(batch_file)  # Update LRU order 更新LRU顺序
            return self._file_cache[batch_file]
        
        # Cache miss, load file data
        # 缓存未命中，加载文件数据
        self._cache_misses += 1
        if self._cache_misses == 1:
            print(f"Starting data file caching...")
        
        # Load entire file to memory
        # 将整个文件加载到内存
        file_data = {}
        with np.load(batch_file) as data:
            file_data['variant_embeddings'] = data['variant_embeddings'].copy()
            file_data['penetrance_estimates'] = data['penetrance_estimates'].copy()
            file_data['varids'] = data['varids'].copy()
            
            # Copy feature data
            # 复制特征数据
            for col in self.annotation_columns + self.gene_columns:
                if col in data:
                    file_data[col] = data[col].copy()
        
        # Cache management: remove LRU file when cache is full
        # 缓存管理：当缓存满时移除LRU文件
        if len(self._file_cache) >= self._max_cache_size:
            if len(self._file_cache) == self._max_cache_size and hasattr(self, '_cache_full_printed') == False:
                print(f"Cache full ({self._max_cache_size} files), starting LRU replacement")
                self._cache_full_printed = True
            oldest_file, _ = self._file_cache.popitem(last=False)
        
        self._file_cache[batch_file] = file_data
        
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
                'embedding': torch.tensor(embedding, dtype=torch.float32).unsqueeze(0),  # Add sequence dimension: [D] -> [1, D]
                'annotation_features': torch.tensor(annotation_features, dtype=torch.float32).unsqueeze(0),  # Add sequence dimension: [D] -> [1, D]
                'gene_features': torch.tensor(gene_features, dtype=torch.float32).unsqueeze(0),  # Add sequence dimension: [D] -> [1, D]
                'label': torch.tensor(label, dtype=torch.float32),
                'varid': varid
            }
            
        except Exception as e:
            # Error handling: return zero vectors as default
            # 错误处理：返回零向量作为默认值
            print(f"Warning: Failed to read sample (file: {batch_file}, index: {sample_idx}): {e}")
            return {
                'embedding': torch.zeros(1, 256, dtype=torch.float32),  # Add sequence dimension: [D] -> [1, D]
                'annotation_features': torch.zeros(1, len(self.annotation_columns), dtype=torch.float32),  # Add sequence dimension: [D] -> [1, D]
                'gene_features': torch.zeros(1, len(self.gene_columns), dtype=torch.float32),  # Add sequence dimension: [D] -> [1, D]
                'label': torch.tensor(0.0, dtype=torch.float32),
                'varid': f"error_{sample_idx}"
            }
    
    def get_cache_info(self):
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics 包含缓存统计信息的字典
        """
        return {
            'cached_files': len(self._file_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{self._cache_hits/(self._cache_hits+self._cache_misses)*100:.1f}%" if (self._cache_hits + self._cache_misses) > 0 else "0%"
        }
    
    def print_cache_summary(self):
        """Print cache summary information.
        
        打印缓存摘要信息。
        """
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = self._cache_hits/(self._cache_hits+self._cache_misses)*100
            print(f"Data cache stats: {len(self._file_cache)}/{len(self.batch_files)} files cached, hit rate: {hit_rate:.1f}%")
    
    
