import copy
import os
import random
import time
from functools import partial, wraps
from typing import Callable, List, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.strategies.ddp import DDPStrategy
from tqdm.auto import tqdm

# Rich library for pretty configuration printing
# 用于美化终端输出，特别是配置信息的显示
from rich import print as rprint
from rich.tree import Tree
from rich.pretty import pprint

from src.data.datamodule import PenetranceDataModule
from src.models.penetrance_net import PenetranceNet

# Enable TensorFloat32 for faster large model training
# 在支持的GPU上启用TensorFloat32以加速训练
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Register OmegaConf resolvers
# 注册自定义配置解析器，用于在配置文件中执行Python表达式
OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

# Custom experiment class for WandbLogger retry mechanism
# 在分布式训练中，只有rank0进程需要与Wandb进行交互，其他进程使用DummyExperiment
class DummyExperiment:
    """Dummy experiment class for no-op operations in non-rank0 processes.
    
    在分布式训练中，非主进程不需要实际记录日志，此类提供空操作以避免错误。
    """

    def nop(self, *args, **kw):
        """No-operation function. 不执行任何操作的函数。"""
        pass

    def __getattr__(self, _):
        """Return no-op function. 返回空操作函数。"""
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        """Support indexing for self.logger.experiment[0].add_image(...) calls.
        
        支持索引访问，用于兼容WandbLogger的API调用。
        """
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        """No-op setter function. 空操作的setter函数。"""
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Decorator: return real experiment object in rank0, dummy in others.
    
    装饰器：在rank0进程中返回真实的实验对象，在其他进程中返回虚拟对象。
    
    Args:
        fn: Function to be decorated 需要装饰的函数
        
    Returns:
        Decorated function 装饰后的函数
    """

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):
    """Custom WandbLogger with retry mechanism and better error handling.
    
    自定义的WandbLogger，增加了重试机制和更好的错误处理能力。
    """

    def __init__(self, *args, **kwargs):
        """Initialize custom WandbLogger.
        
        Args:
            *args: Positional arguments 位置参数
            **kwargs: Keyword arguments 关键字参数
        """
        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        """Get real wandb experiment object.
        
        获取真实的wandb实验对象，包含重试机制以处理网络不稳定等问题。
        
        Returns:
            wandb.Run: wandb experiment run object Wandb实验运行对象
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                # 当前实例中已存在wandb进程
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # Attach to referenced wandb process
                # 附加到指定的wandb进程
                self._experiment = wandb._attach(attach_id)
            else:
                # Create new wandb process with retry mechanism
                # 创建新的wandb进程，包含重试机制
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # Define default x-axis
                # 定义默认的x轴指标
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class TrainingPenetranceNet(PenetranceNet):
    """PenetranceNet for training with all training-related logic.
    
    专为训练设计的PenetranceNet扩展类，包含了训练过程中的损失计算和优化器配置。
    """

    def __init__(self, cfg):
        """Initialize training model.
        
        Args:
            cfg: Configuration object 配置对象，包含模型和训练参数
        """
        # Disable JIT profiling executor for memory efficiency and speed
        # 禁用JIT性能分析执行器以提高内存效率和速度
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__(cfg)
        self.cfg = cfg
        self.criterion = nn.MSELoss()
        
        # PyTorch Lightning bug fix: ensure setup is called only once
        # PyTorch Lightning bug修复：确保setup方法只被调用一次
        self._has_setup = False

    def setup(self, stage=None):
        """Model setup stage.
        
        Args:
            stage: Training stage identifier 训练阶段标识符，如'fit', 'validate', 'test'
        """
        # Prevent duplicate model setup in DDP training causing memory imbalance
        # 防止在DDP训练中重复设置模型导致内存不平衡
        if self._has_setup:
            return
        else:
            # Call parent setup method to create network layers
            # 调用父类setup方法创建网络层
            super().setup(stage)
            self._has_setup = True

    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch: Batch data 批次数据
            batch_idx: Batch index 批次索引
            
        Returns:
            Training loss 训练损失值
        """
        prediction, _ = self(batch)
        loss = self.criterion(prediction.squeeze(), batch['label'])
        
        # Log batch size for logging
        # 记录批次大小用于日志记录
        batch_size = batch['embedding'].shape[0]
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: Batch data 批次数据
            batch_idx: Batch index 批次索引
            
        Returns:
            Validation loss 验证损失值
        """
        prediction, _ = self(batch)
        loss = self.criterion(prediction.squeeze(), batch['label'])
        
        # Log batch size for logging
        # 记录批次大小用于日志记录
        batch_size = batch['embedding'].shape[0]
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step.
        
        Args:
            batch: Batch data 批次数据
            batch_idx: Batch index 批次索引
            
        Returns:
            Test loss 测试损失值
        """
        prediction, _ = self(batch)
        loss = self.criterion(prediction.squeeze(), batch['label'])
        
        # Log batch size for logging
        # 记录批次大小用于日志记录
        batch_size = batch['embedding'].shape[0]
        self.log('test_loss', loss, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler 包含优化器和调度器的字典
        """
        # Ensure model setup is complete
        # 确保模型设置完成
        if not self._has_setup:
            self.setup()
            
        # Get optimizer parameters from configuration
        # 从配置中获取优化器参数
        optimizer_name_map = {
            'adamw': 'AdamW',
            'adam': 'Adam',
            'sgd': 'SGD'
        }
        optimizer_name = optimizer_name_map.get(self.hparams.optimizer._name_.lower(), self.hparams.optimizer._name_)
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
            betas=self.hparams.optimizer.betas
        )

        # Configure learning rate scheduler
        # 配置学习率调度器
        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler._name_ == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.scheduler.mode,
                factor=self.hparams.scheduler.factor,
                patience=self.hparams.scheduler.patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.hparams.train.monitor
                }
            }
        else:
            # Default to cosine annealing scheduler
            # 默认使用余弦退火调度器
            max_epochs = 100  # Default value
            if hasattr(self.hparams, 'trainer') and hasattr(self.hparams.trainer, 'max_epochs'):
                max_epochs = self.hparams.trainer.max_epochs
            elif hasattr(self, 'trainer') and hasattr(self.trainer, 'max_epochs'):
                max_epochs = self.trainer.max_epochs
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }


def create_trainer(config, **kwargs):
    """Create PyTorch Lightning trainer.
    
    创建PyTorch Lightning训练器，配置日志记录、回调函数和分布式训练策略。
    
    Args:
        config: Configuration object 配置对象
        **kwargs: Other keyword arguments 其他关键字参数
        
    Returns:
        PyTorch Lightning trainer instance PyTorch Lightning训练器实例
    """
    callbacks: List[pl.Callback] = []
    logger = None

    # Configure WandB logging
    # 配置WandB日志记录
    if config.get("wandb") is not None:
        # Pass wandb.init(config=) parameter to log hyperparameters
        # 传递wandb.init(config=)参数以记录超参数
        import wandb

        logger = CustomWandbLogger(
            config=OmegaConf.to_container(config, resolve=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Configure PyTorch Lightning callbacks
    # 配置PyTorch Lightning回调函数
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            # Create callback instances based on callback names
            # 根据回调函数名称创建回调实例
            if _name_ == "model_checkpoint":
                callbacks.append(pl.callbacks.ModelCheckpoint(**callback))
            elif _name_ == "early_stopping":
                callbacks.append(pl.callbacks.EarlyStopping(**callback))
            elif _name_ == "learning_rate_monitor":
                callbacks.append(pl.callbacks.LearningRateMonitor(**callback))

    # Auto-configure DDP strategy
    # 自动配置DDP策略用于多GPU训练
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices may be list like [1, 3]
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    # Initialize PyTorch Lightning trainer
    # 初始化PyTorch Lightning训练器
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)    

    return trainer


def train(config):
    """Main training function.
    
    主训练函数，负责整个训练流程的执行。
    
    Args:
        config: Configuration object 配置对象
    """
    # Set random seed for reproducibility
    # 设置随机种子以确保实验可重现性
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    
    # Print training configuration
    # 打印训练配置
    print("Training configuration:")
    config_tree = config_to_tree(config)
    rprint(config_tree)
    
    # Create trainer
    # 创建训练器
    trainer = create_trainer(config)
    
    # Initialize data module
    # 初始化数据模块
    dm = PenetranceDataModule(config)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Initialize model
    # 初始化模型
    model = TrainingPenetranceNet(config.model)

    # Load pretrained model (if specified)
    # 加载预训练模型（如果指定）
    if config.train.get("pretrained_model_path", None) is not None:
        model = TrainingPenetranceNet.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )

    # Run initial validation round (for debugging and tuning)
    # 运行初始验证轮次（用于调试和调优）
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    # Start training
    # 开始训练
    if config.train.ckpt is not None:
        trainer.fit(model, dm, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model, dm)
    
    # Run test (if configured)
    # 运行测试（如果配置）
    if config.train.test:
        trainer.test(model, dm)


def config_to_tree(config, tree=None, name="Configuration"):
    """Convert OmegaConf configuration to rich tree structure for visualization.
    
    Args:
        config: Configuration object (DictConfig or ListConfig)
        tree: Tree structure object (optional)
        name: Root node name for tree structure
        
    Returns:
        Rich tree structure object
    """
    from omegaconf import DictConfig, ListConfig
    
    if tree is None:
        tree = Tree(name)
    
    if isinstance(config, DictConfig):
        # Handle dictionary type configuration
        for key, value in config.items():
            if isinstance(value, DictConfig):
                # Recursively handle nested configuration
                subtree = tree.add(f"[bold]{key}[/bold]")
                config_to_tree(value, subtree, key)
            elif isinstance(value, ListConfig):
                # Handle list configuration
                subtree = tree.add(f"[bold]{key}[/bold]")
                for i, item in enumerate(value):
                    if isinstance(item, (DictConfig, ListConfig)):
                        item_tree = subtree.add(f"[cyan]{i}[/cyan]")
                        config_to_tree(item, item_tree, str(i))
                    else:
                        subtree.add(f"[cyan]{i}[/cyan] = {item}")
            else:
                # Leaf node
                tree.add(f"[cyan]{key}[/cyan] = {value}")
    elif isinstance(config, ListConfig):
        # Handle list configuration
        for i, item in enumerate(config):
            if isinstance(item, (DictConfig, ListConfig)):
                item_tree = tree.add(f"[cyan]{i}[/cyan]")
                config_to_tree(item, item_tree, str(i))
            else:
                tree.add(f"[cyan]{i}[/cyan] = {item}")
    else:
        # Leaf node
        tree.label = f"{tree.label} = {config}"
    
    return tree


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(config: OmegaConf):
    """Main function entry point.
    
    Args:
        config: Configuration object
    """
    train(config)


if __name__ == "__main__":
    # Windows multiprocessing support
    import multiprocessing
    multiprocessing.freeze_support()
    main()
