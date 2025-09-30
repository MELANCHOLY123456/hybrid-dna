import torch
import torch.nn as nn
import json
import os
import re
from typing import Optional
from standalone_hyenadna import HyenaDNAModel


def inject_substring(orig_str):
    """处理使用和不使用梯度检查点的模型之间的键匹配问题
    
    该函数用于解决不同版本HyenaDNA模型权重键名不一致的问题，
    通过正则表达式替换来统一键名格式。
    
    Args:
        orig_str (str): 原始键名字符串
        
    Returns:
        str: 处理后的键名字符串
    """
    # 修改 mixer 键名：添加.layer后缀
    pattern = r"\.mixer"
    injection = ".mixer.layer"
    modified_string = re.sub(pattern, injection, orig_str)

    # 修改 mlp 键名：添加.layer后缀
    pattern = r"\.mlp"
    injection = ".mlp.layer"
    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string


def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """将预训练权重加载到初始化模型的状态字典中
    
    该函数处理不同模型版本间的权重键名差异，确保权重能够正确加载。
    
    Args:
        scratch_dict (dict): 初始化模型的状态字典
        pretrained_dict (dict): 预训练模型的权重字典
        checkpointing (bool): 是否使用梯度检查点，默认为False
        
    Returns:
        dict: 更新后的状态字典
        
    Raises:
        Exception: 当键名不匹配时抛出异常
    """
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # 状态字典前缀不同，添加 '.model' 前缀
            key_loaded = 'model.' + key
            # 如果使用梯度检查点，需要进一步处理键名
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('状态字典中的键不匹配!')
    return scratch_dict


class HyenaDNAFeatureExtractor(nn.Module):
    """HyenaDNA特征提取器
    
    该类封装了HyenaDNA模型，用于从DNA序列中提取token级别的embedding特征。
    支持加载预训练权重和冻结模型参数等功能。
    """

    def __init__(self, checkpoint_path: str, model_name: str, freeze: bool = False,
                 device: str = 'cuda', use_head: bool = False, n_classes: int = 2):
        """初始化HyenaDNA特征提取器
        
        Args:
            checkpoint_path (str): 模型检查点根目录路径
            model_name (str): 模型名称
            freeze (bool): 是否冻结模型参数，默认为False
            device (str): 运行设备，'cuda'或'cpu'，默认为'cuda'
            use_head (bool): 是否使用分类头，默认为False
            n_classes (int): 分类类别数，默认为2
        """
        super().__init__()

        # Build complete model path
        pretrained_model_path = os.path.join(checkpoint_path, model_name)

        # Validate model path exists
        if not os.path.isdir(pretrained_model_path):
            raise ValueError(f"Model path does not exist: {pretrained_model_path}")

        # Load model configuration file
        config_path = os.path.join(pretrained_model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 初始化空白模型实例
        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)

        # 加载预训练权重文件
        weights_path = os.path.join(pretrained_model_path, 'weights.ckpt')
        loaded_ckpt = torch.load(weights_path, map_location=torch.device(device), weights_only=False)

        # 检查是否使用了梯度检查点技术
        checkpointing = config.get("checkpoint_mixer", False)

        # 将预训练权重加载到模型中
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'],
                                  checkpointing=checkpointing)
        scratch_model.load_state_dict(state_dict)

        print(f"成功加载预训练权重: {model_name}")

        # 设置模型属性
        self.model = scratch_model
        self.model.to(device)
        self.model.eval()

        # 根据配置冻结模型参数
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, tokens):
        """前向传播，提取序列特征
        
        Args:
            tokens (torch.Tensor): 输入的token序列张量
                - 形状: [batch_size, seq_len] 或 [seq_len]
                
        Returns:
            torch.Tensor: 提取的token级别embedding特征
                - 形状: [batch_size, seq_len, hidden_dim]
        """
        # 处理单序列输入情况：[seq_len] -> [1, seq_len]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # 在推理模式下执行前向传播
        with torch.inference_mode():
            outputs = self.model(tokens)  # 输出形状: [batch_size, seq_len, hidden_dim]

        return outputs