import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
import numpy as np
from src.data.genome_processor import GenomeProcessor
from src.components.feature_extractors import HyenaDNAFeatureExtractor
from src.utils.tokenization import DNATokenizer
import logging
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Gene variant data preprocessing main function.
    
    Extracts sequence features from raw variant data and generates embedding
    data for training. Processing pipeline includes: data loading, sequence
    extraction, feature encoding, embedding extraction, and data saving.
    
    Args:
        cfg: Configuration object
    """
    # 加载变异数据文件
    variants_path = f"{cfg.data.data_dir}/{cfg.data.variants_file}"
    variants_df = pd.read_csv(variants_path, sep='\t', low_memory=False)
    logger.info(f"成功加载变异数据，共 {len(variants_df)} 条记录")
    logger.info(f"数据列: {list(variants_df.columns[:10])}...（显示前10列）")
    logger.info(f"总共 {len(variants_df.columns)} 列")
    
    # 查找可能的标签列（包含penetrance、label或target关键词的列）
    possible_labels = [col for col in variants_df.columns if 'penetrance' in col.lower() or 'label' in col.lower() or 'target' in col.lower()]
    if possible_labels:
        logger.info(f"找到可能的标签列: {possible_labels}")
    else:
        logger.info("未找到明显的标签列，将查看所有列名")
        logger.info(f"所有列名: {list(variants_df.columns)}")

    # 初始化基因组处理器，用于提取DNA序列
    genome_path = f"{cfg.data.data_dir}/{cfg.data.reference_genome}"
    genome_processor = GenomeProcessor(genome_path)

    # 初始化HyenaDNA特征提取器，用于生成序列embedding
    logger.info(f"正在加载 HyenaDNA 模型: hyenadna-small-32k-seqlen")
    feature_extractor = HyenaDNAFeatureExtractor(
        checkpoint_path="./checkpoints",
        model_name="hyenadna-small-32k-seqlen",
        freeze=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 初始化DNA序列分词器
    tokenizer = DNATokenizer()

    # 确保输出目录存在
    os.makedirs(f"{cfg.data.data_dir}/processed", exist_ok=True)

    # 设置批次大小，用于分批处理数据
    batch_size = cfg.data.batch_size if hasattr(cfg.data, 'batch_size') else 1000
    total_batches = (len(variants_df) + batch_size - 1) // batch_size

    # 初始化存储结构，用于记录处理失败的变异
    all_failed_variants = []

    logger.info(f"开始分批提取序列embedding，共 {total_batches} 批，每批 {batch_size} 条记录...")

    def save_batch_results(batch_data):
        """保存批次处理结果到磁盘
        
        Args:
            batch_data (tuple): 包含批次数据的元组
            
        Returns:
            bool: 保存是否成功
        """
        batch_idx, embeddings_array, varids, batch_final_df, batch_output_path = batch_data
        try:
            # 生成NPZ文件路径（替换.h5为.npz）
            embedding_path = batch_output_path.replace('.h5', '.npz')
            
            # 准备要保存的数据字典
            save_dict = {
                'embeddings': embeddings_array,
                'varids': np.array(varids)
            }
            
            # 添加所有数据列到NPZ文件中
            for col in batch_final_df.columns:
                if col != 'varid':
                    save_dict[col] = batch_final_df[col].values
            
            # 保存到NPZ压缩文件
            np.savez_compressed(embedding_path, **save_dict)
            
            logger.info(f"批次 {batch_idx + 1} 保存完成！数据已保存到 {embedding_path}")
            return True
        except Exception as e:
            logger.error(f"保存批次 {batch_idx + 1} 时出错: {e}")
            return False

    # 创建线程池用于并发保存，提高处理效率
    save_executor = ThreadPoolExecutor(max_workers=2)
    save_futures = []

    # 分批处理数据
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(variants_df))
        batch_df = variants_df.iloc[start_idx:end_idx]

        logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批 ({start_idx}-{end_idx})")

        # 初始化批次存储结构
        embeddings = []
        varids = []
        failed_variants = []

        # 逐个处理批次中的变异记录
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"批次 {batch_idx + 1}"):
            varid = row['varid']

            # 从变异ID中解析染色体和位置信息
            try:
                chrom, pos_str, ref, alt = varid.split('-')
                pos = int(pos_str)
            except ValueError:
                logger.warning(f"无法解析变异ID: {varid}，跳过此变异")
                failed_variants.append(varid)
                continue

            # 提取以变异为中心的DNA序列
            sequence = genome_processor.get_centered_sequence(
                chrom, pos, cfg.data.seq_len // 2
            )

            # 处理序列提取失败的情况
            if sequence is None or len(sequence) == 0:
                logger.warning(f"无法提取变异 {varid} 的序列，使用零向量")
                embedding = np.zeros((cfg.data.seq_len, 256), dtype=np.float32)
                failed_variants.append(varid)
            else:
                # 确保序列长度符合要求
                if len(sequence) != cfg.data.seq_len:
                    logger.warning(f"变异 {varid} 的序列长度不正确: {len(sequence)} != {cfg.data.seq_len}")
                    # 截断或填充序列至指定长度
                    if len(sequence) > cfg.data.seq_len:
                        sequence = sequence[:cfg.data.seq_len]
                    else:
                        sequence = sequence + 'N' * (cfg.data.seq_len - len(sequence))

                # 将DNA序列编码为整数token序列
                tokens = tokenizer.encode(sequence)
                tokens_tensor = torch.tensor(tokens, dtype=torch.long)

                # 将token张量移动到与模型相同的设备
                device = next(feature_extractor.parameters()).device
                tokens_tensor = tokens_tensor.to(device)

                # 使用HyenaDNA模型提取序列embedding
                try:
                    with torch.no_grad():
                        embedding_tensor = feature_extractor(tokens_tensor.unsqueeze(0))
                        # 移除batch维度并转换为numpy数组
                        embedding = embedding_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                except Exception as e:
                    logger.error(f"提取变异 {varid} embedding时出错: {e}")
                    embedding = np.zeros((cfg.data.seq_len, 256), dtype=np.float32)
                    failed_variants.append(varid)

            # 存储处理结果
            embeddings.append(embedding)
            varids.append(varid)

        # 准备保存数据
        model_name = "hyenadna-small-32k-seqlen"
        batch_output_path = f"{cfg.data.data_dir}/processed/variants_with_embeddings_{model_name}_batch_{batch_idx}.npz"
        
        # 将embedding转换为numpy数组以便存储
        embeddings_array = np.array(embeddings)
        
        # 创建包含变异ID的DataFrame
        embedding_df = pd.DataFrame({
            'varid': varids
        })

        # 合并回原始数据，只保留训练需要的列
        available_cols = batch_df.columns.tolist()
        required_cols = ['varid'] + cfg.data.annotation_columns + cfg.data.gene_columns
        
        # 查找标签列
        label_candidates = ['penetrance_estimates', 'penetrance_estimate', 'penetrance', 'label', 'target']
        label_col = None
        for candidate in label_candidates:
            if candidate in available_cols:
                label_col = candidate
                break
        
        # 添加标签列到需要保留的列中
        if label_col:
            required_cols.append(label_col)
        
        # 只选择存在的列，并转换数据类型
        numeric_cols = [col for col in required_cols if col in available_cols]
        batch_subset = batch_df[numeric_cols].copy()
        
        # 转换字符串列为数值类型
        for col in batch_subset.columns:
            if col != 'varid':
                if batch_subset[col].dtype == 'object':
                    try:
                        batch_subset[col] = pd.to_numeric(batch_subset[col], errors='coerce')
                    except:
                        pass
        
        # 验证标签列是否存在
        if label_col is None:
            logger.warning("未找到外显率标签列，请检查数据文件")
        batch_final_df = pd.merge(batch_subset, embedding_df, on='varid')
        
        # 提交保存任务到后台线程池
        batch_data = (batch_idx, embeddings_array, varids, batch_final_df, batch_output_path)
        future = save_executor.submit(save_batch_results, batch_data)
        save_futures.append(future)
        
        logger.info(f"批次 {batch_idx + 1} embedding提取完成，已提交保存任务")

        # 收集处理失败的变异
        all_failed_variants.extend(failed_variants)

        # 清理内存
        del embeddings, varids, embedding_df, batch_final_df, embeddings_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 限制并发保存任务数量，避免内存占用过高
        if len(save_futures) >= 2:
            completed_future = save_futures.pop(0)
            completed_future.result()

    # 等待所有保存任务完成
    logger.info("等待所有保存任务完成...")
    for future in as_completed(save_futures):
        try:
            result = future.result()
            if not result:
                logger.warning("某个保存任务失败")
        except Exception as e:
            logger.error(f"保存任务异常: {e}")
    
    save_executor.shutdown(wait=True)
    logger.info("所有批次保存完成")

    # 合并所有批次的结果文件
    logger.info("合并所有批次结果...")
    model_name = "hyenadna-small-32k-seqlen"
    
    # 收集所有批次文件
    batch_files = []
    for batch_idx in range(total_batches):
        batch_file = f"{cfg.data.data_dir}/processed/variants_with_embeddings_{model_name}_batch_{batch_idx}.npz"
        if os.path.exists(batch_file):
            batch_files.append(batch_file)
    
    # 验证是否存在批次文件
    if not batch_files:
        logger.error("没有找到任何批次文件！")
        return
    
    logger.info(f"找到 {len(batch_files)} 个批次文件，开始合并...")
    
    # 逐个读取批次文件并合并数据
    all_varids = []
    all_embeddings_list = []
    all_data_dict = {}
    
    for i, batch_file in enumerate(batch_files):
        logger.info(f"读取批次文件 {i+1}/{len(batch_files)}: {batch_file}")
        batch_data = np.load(batch_file)
        
        # 收集变异ID和embeddings
        all_varids.extend(batch_data['varids'])
        all_embeddings_list.append(batch_data['embeddings'])
        
        # 收集其他数据列
        for key in batch_data.keys():
            if key not in ['varids', 'embeddings']:
                if key not in all_data_dict:
                    all_data_dict[key] = []
                all_data_dict[key].append(batch_data[key])
        
        batch_data.close()
    
    # 合并所有embeddings数组
    logger.info("合并embedding数组...")
    all_embeddings_array = np.concatenate(all_embeddings_list, axis=0)
    del all_embeddings_list
    
    # 合并其他数据列
    for key in all_data_dict:
        all_data_dict[key] = np.concatenate(all_data_dict[key], axis=0)
    
    # 保存完整的处理后数据到NPZ格式
    output_path = f"{cfg.data.data_dir}/processed/variants_with_embeddings_{model_name}.npz"
    
    # 准备要保存的数据字典
    save_dict = {
        'embeddings': all_embeddings_array,
        'varids': np.array(all_varids)
    }
    save_dict.update(all_data_dict)
    
    # 保存到NPZ压缩文件
    logger.info("保存合并后的完整数据...")
    np.savez_compressed(output_path, **save_dict)

    logger.info(f"预处理完成！完整数据已保存到 {output_path}")
    logger.info(f"总数据量: {len(all_varids)} 条")
    logger.info(f"每条embedding形状: {all_embeddings_array.shape[1:] if len(all_embeddings_array) > 0 else 'N/A'}")
    
    # 处理失败变异的记录和保存
    if all_failed_variants:
        logger.warning(f"有 {len(all_failed_variants)} 条变异处理失败")
        # 保存失败变异的列表
        failed_path = f"{cfg.data.data_dir}/processed/failed_variants.txt"
        with open(failed_path, 'w') as f:
            for varid in all_failed_variants:
                f.write(f"{varid}\n")
        logger.info(f"失败变异列表已保存到 {failed_path}")


if __name__ == "__main__":
    main()
