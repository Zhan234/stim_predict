"""基于GRPO（Group Relative Policy Optimization）的超图权重预测方法"""

import stim
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Optional
import correlation
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

from .base import BasePredictor


class GRPOPredictor(BasePredictor):
    """
    基于GRPO（Group Relative Policy Optimization）的预测器
    
    GRPO是一种高效的强化学习算法，基于PPO改进，去除critic模型，
    转而通过采样一组输出的分数来估计基线，从而减少训练资源。
    
    核心特点:
    1. 去除critic模型，不需要值函数
    2. 使用组内输出的相对奖励作为优势估计
    3. 支持结果监督（Outcome Supervision）和过程监督（Process Supervision）
    4. 使用KL散度正则化防止策略偏离参考模型
    
    参考: GRPO论文 - Group Relative Policy Optimization
    """
    
    def __init__(self,
                 learning_rate: float = 1e-6,
                 group_size: int = 64,
                 epochs: int = 100,
                 clip_ratio: float = 0.2,
                 kl_coef: float = 0.04,
                 max_grad_norm: float = 0.5,
                 supervision_mode: str = 'outcome',
                 use_gpu: bool = True,
                 num_workers: int = 8,
                 eval_subset_size: int = 10000):
        """
        初始化GRPO预测器
        
        Args:
            learning_rate: 学习率（默认1e-6，GRPO论文推荐）
            group_size: 组大小G（默认64，GRPO论文推荐）
            epochs: 训练轮数
            clip_ratio: PPO重要性比率裁剪阈值（类似PPO）
            kl_coef: KL散度系数β（默认0.04）
            max_grad_norm: 梯度裁剪的最大范数
            supervision_mode: 监督模式，'outcome'（结果监督）或'process'（过程监督）
            use_gpu: 是否使用GPU
            num_workers: 并行解码的工作线程数
            eval_subset_size: 每次评估使用的样本子集大小（0表示使用全部）
        """
        super().__init__(name="grpo")
        
        # 超参数
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.supervision_mode = supervision_mode
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_workers = num_workers
        self.eval_subset_size = eval_subset_size
        
        # 设备设置
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU")
        
        # 策略参数（使用PyTorch张量）
        self.policy_mean = None
        self.policy_std = None
        
        # 旧策略参数（用于计算importance ratio）
        self.old_policy_mean = None
        self.old_policy_std = None
        
        # 参考策略（用于KL散度正则化）
        self.reference_policy_mean = None
        self.reference_policy_std = None
        
        # Tanner图
        self.tanner_graph = None
        
        # 训练历史
        self.training_history = {
            'rewards': [],
            'ler': [],
            'policy_loss': [],
            'kl_loss': [],
            'advantages': []
        }
    
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        """
        使用GRPO训练预测器
        
        Args:
            circuit: Stim电路对象
            detector_samples: 探测器采样数据
            **kwargs: 其他参数，可包含:
                - observables: 观测量数据
                - decoder_type: 解码器类型，默认'pymatching'
                
        Returns:
            训练结果字典
        """
        observables = kwargs.get('observables', None)
        decoder_type = kwargs.get('decoder_type', 'pymatching')
        
        # 获取原始DEM和Tanner图
        dem_origin = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True
        )
        self.tanner_graph = correlation.TannerGraph(dem_origin)
        
        # 初始化策略参数
        n_params = len(self.tanner_graph.hyperedge_probs)
        
        # 使用平坦的小概率初始化，避免依赖DEM先验
        init_probs = np.ones(n_params) * 1e-3
        init_probs = np.clip(init_probs, 1e-10, 0.49)
        
        # 转换为PyTorch张量（需要requires_grad=True以便优化）
        self.policy_mean = nn.Parameter(
            torch.tensor(
                np.log(init_probs), 
                dtype=torch.float32,
                device=self.device
            )
        )
        self.policy_std = nn.Parameter(
            torch.tensor(
                np.ones(n_params) * 0.5,
                dtype=torch.float32,
                device=self.device
            )
        )
        
        # 初始化旧策略和参考策略
        self.old_policy_mean = self.policy_mean.clone().detach()
        self.old_policy_std = self.policy_std.clone().detach()
        self.reference_policy_mean = self.policy_mean.clone().detach()
        self.reference_policy_std = self.policy_std.clone().detach()
        
        # 优化器
        optimizer = torch.optim.Adam(
            [self.policy_mean, self.policy_std],
            lr=self.learning_rate
        )
        
        print(f"开始GRPO训练，共 {self.epochs} 轮，组大小 {self.group_size}")
        print(f"监督模式: {self.supervision_mode}")
        print(f"并行工作线程: {self.num_workers}, 评估样本子集大小: {self.eval_subset_size if self.eval_subset_size > 0 else '全部'}")
        
        # GRPO训练循环
        for epoch in range(self.epochs):
            # 1. 从当前策略采样一组输出（group）
            noise, log_probs_group, probs_group = self._sample_group(self.group_size)
            
            # 2. 并行评估每个候选的奖励
            probs_group_np = probs_group.detach().cpu().numpy()
            rewards, lers = self._evaluate_batch_parallel(
                probs_group_np,
                detector_samples,
                observables,
                decoder_type
            )
            
            rewards = np.array(rewards)
            lers = np.array(lers)
            
            # 3. 计算组内优势（GRPO核心：使用组内相对奖励）
            advantages = self._compute_group_advantages(rewards)
            
            # 4. 保存旧策略（在更新前）
            self.old_policy_mean = self.policy_mean.clone().detach()
            self.old_policy_std = self.policy_std.clone().detach()
            
            # 5. GRPO策略更新
            policy_loss, kl_loss = self._update_policy_grpo(
                noise,
                log_probs_group,
                advantages,
                optimizer
            )
            
            # 记录训练历史
            self.training_history['rewards'].append(np.mean(rewards))
            self.training_history['ler'].append(np.mean(lers))
            self.training_history['policy_loss'].append(policy_loss)
            self.training_history['kl_loss'].append(kl_loss)
            self.training_history['advantages'].append(np.mean(advantages))
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"Mean Reward={np.mean(rewards):.3f}, "
                    f"Mean LER={np.mean(lers):.6f}, "
                    f"Policy Loss={policy_loss:.6e}, "
                    f"KL Loss={kl_loss:.6e}"
                )
        
        # 使用最终策略的均值作为预测（去除梯度以便转换为numpy）
        final_probs = torch.exp(self.policy_mean.detach()).cpu().numpy()
        final_hyperedge_probs = {}
        for i, hyperedge in enumerate(self.tanner_graph.hyperedge_probs.keys()):
            final_hyperedge_probs[hyperedge] = final_probs[i]
        
        self.hyperedge_probs = final_hyperedge_probs
        self.trained = True
        
        return {
            'hyperedge_probs': final_hyperedge_probs,
            'training_history': self.training_history,
            'final_mean_reward': self.training_history['rewards'][-1],
            'final_mean_ler': self.training_history['ler'][-1]
        }
    
    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        """
        预测超边概率
        
        Args:
            circuit: Stim电路对象
            
        Returns:
            超边到概率的映射字典
        """
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self.hyperedge_probs
    
    def _sample_group(self, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从当前策略采样一组输出
        
        Args:
            group_size: 组大小G
            
        Returns:
            (noise, log_probs_group_detached, probs_group): 噪声张量、对数概率张量（无梯度）
            和概率张量，形状 (group_size, n_params)
        """
        n_params = len(self.policy_mean)
        
        # 从高斯策略采样（在对数空间）
        # 只采样噪声，梯度在_update_policy_grpo中通过重新计算获得
        noise = torch.randn(
            (group_size, n_params),
            device=self.device,
            dtype=torch.float32
        )
        
        # 计算log_probs用于评估（不需要梯度）
        with torch.no_grad():
            log_probs_group = self.policy_mean.unsqueeze(0) + noise * self.policy_std.unsqueeze(0)
            probs_group = torch.clamp(torch.exp(log_probs_group), min=1e-10, max=1.0)
        
        # 训练时会重新基于当前 μ/σ 计算 log prob，因此这里返回 detach 的采样值即可
        return noise, log_probs_group.detach(), probs_group
    
    def _evaluate_batch_parallel(self,
                                  probs_batch: np.ndarray,
                                  detector_samples: np.ndarray,
                                  observables: Optional[np.ndarray],
                                  decoder_type: str) -> Tuple[List[float], List[float]]:
        """
        并行评估一批候选参数的LER
        
        Args:
            probs_batch: 概率参数批次，形状 (batch_size, n_params)
            detector_samples: 探测器采样数据
            observables: 观测量数据
            decoder_type: 解码器类型
            
        Returns:
            (rewards, lers): 奖励列表和LER列表
        """
        # 子采样以加速评估
        if self.eval_subset_size > 0 and len(detector_samples) > self.eval_subset_size:
            indices = np.random.choice(len(detector_samples), self.eval_subset_size, replace=False)
            eval_samples = detector_samples[indices]
            eval_observables = observables[indices] if observables is not None else None
        else:
            eval_samples = detector_samples
            eval_observables = observables
        
        hyperedge_keys = list(self.tanner_graph.hyperedge_probs.keys())
        
        def eval_single(probs: np.ndarray) -> Tuple[float, float]:
            """评估单个候选"""
            hyperedge_probs = {h: probs[j] for j, h in enumerate(hyperedge_keys)}
            dem = self._build_dem_from_hyperedge_probs(hyperedge_probs)
            ler = self._evaluate_ler(dem, eval_samples, eval_observables, decoder_type)
            reward = -np.log10(max(ler, 1e-10))
            return reward, ler
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(eval_single, probs_batch))
        
        rewards = [r[0] for r in results]
        lers = [r[1] for r in results]
        
        return rewards, lers
    
    def _compute_group_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """
        计算组内优势（GRPO核心）
        
        优势计算基于组内奖励的标准化（减去均值、除以标准差）
        
        Args:
            rewards: 组内奖励数组，形状 (group_size,)
            
        Returns:
            优势数组，形状 (group_size,)
        """
        # GRPO: 使用组内相对奖励作为优势
        # 对于结果监督：所有token的优势为标准化奖励
        # A_{i,t} = (r_i - mean(r)) / std(r)
        
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards) + 1e-8  # 避免除零
        
        advantages = (rewards - rewards_mean) / rewards_std
        # 避免极端样本主导梯度
        advantages = np.clip(advantages, -5.0, 5.0)
        
        return advantages
    
    def _update_policy_grpo(self,
                           noise: torch.Tensor,
                           log_probs_group_detached: torch.Tensor,
                           advantages: np.ndarray,
                           optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """
        使用GRPO更新策略
        
        实现GRPO目标函数（论文Equation 3）:
        J_GRPO(θ) = E[1/G * Σ_i (π_θ/π_θ_old * A_i - β * D_KL[π_θ || π_ref])]
        
        Args:
            noise: 采样噪声，形状 (group_size, n_params)
            advantages: 优势数组，形状 (group_size,)
            optimizer: 优化器
            
        Returns:
            (policy_loss, kl_loss): 策略损失和KL散度损失
        """
        group_size, n_params = noise.shape
        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )
        
        # 计算当前策略下采样到 log_probs_group_detached 的对数概率
        z_current = (log_probs_group_detached - self.policy_mean.unsqueeze(0)) / (self.policy_std.unsqueeze(0) + 1e-10)
        log_prob_current = -0.5 * torch.sum(z_current ** 2, dim=1) - torch.sum(torch.log(self.policy_std + 1e-10))
        
        # 计算旧策略的对数概率
        z_old = (log_probs_group_detached - self.old_policy_mean.unsqueeze(0)) / (self.old_policy_std.unsqueeze(0) + 1e-10)
        log_prob_old = -0.5 * torch.sum(z_old ** 2, dim=1) - torch.sum(torch.log(self.old_policy_std + 1e-10))
        
        # 计算importance ratio: π_θ / π_θ_old
        log_ratio = log_prob_current - log_prob_old
        importance_ratio = torch.exp(torch.clamp(log_ratio, -10, 10))
        
        # PPO clipping
        importance_ratio_clipped = torch.clamp(
            importance_ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio
        )
        
        # 计算KL散度
        # D_KL[π_θ || π_ref] = E_π_θ[log(π_θ/π_ref)]
        z_ref = (log_probs_group_detached - self.reference_policy_mean.unsqueeze(0)) / (self.reference_policy_std.unsqueeze(0) + 1e-10)
        log_prob_ref = -0.5 * torch.sum(z_ref ** 2, dim=1) - torch.sum(torch.log(self.reference_policy_std + 1e-10))
        
        log_ratio_kl = log_prob_current - log_prob_ref
        kl_div = log_ratio_kl
        
        # GRPO目标函数
        # J = E[(π_θ/π_θ_old * A) - β * D_KL]
        # 使用clipped importance ratio
        # advantages_tensor形状为(group_size,)，需要扩展到每个样本
        policy_objective = importance_ratio_clipped * advantages_tensor
        kl_penalty = self.kl_coef * kl_div
        
        # 目标函数（最大化，所以取负号作为损失）
        loss = -torch.mean(policy_objective - kl_penalty)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [self.policy_mean, self.policy_std],
            self.max_grad_norm
        )
        
        optimizer.step()
        
        # 计算用于监控的损失值
        policy_loss = -torch.mean(policy_objective).item()
        kl_loss = torch.mean(kl_penalty).item()
        
        return policy_loss, kl_loss
    
    def _build_dem_from_hyperedge_probs(self, hyperedge_probs: Dict[Tuple, float]) -> stim.DetectorErrorModel:
        """
        从超边概率构建DEM
        
        Args:
            hyperedge_probs: 超边概率字典
            
        Returns:
            DetectorErrorModel
        """
        dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in hyperedge_probs.items():
            if prob > 0:
                prob = np.clip(prob, 1e-10, 1.0)  # 确保概率有效
                
                decompose = self.tanner_graph.stim_decompose[hyperedge]
                targets = []
                
                for line_i in range(len(decompose)):
                    h = decompose[line_i]
                    t = self.tanner_graph.hyperedge_frames
                    
                    targets += [stim.DemTarget(f"D{id_index}") for id_index in h]
                    targets += [stim.DemTarget(f"L{id_index}") for id_index in t[h]]
                    
                    if line_i != len(decompose) - 1:
                        targets.append(stim.DemTarget("^"))
                
                instruction = stim.DemInstruction("error", [prob], targets)
                dem.append(instruction)
        
        return dem
    
    def _evaluate_ler(self,
                     dem: stim.DetectorErrorModel,
                     detector_samples: np.ndarray,
                     observables: Optional[np.ndarray],
                     decoder_type: str = 'pymatching') -> float:
        """
        评估给定DEM的逻辑错误率
        
        Args:
            dem: DetectorErrorModel
            detector_samples: 探测器采样
            observables: 观测量真值
            decoder_type: 解码器类型
            
        Returns:
            逻辑错误率
        """
        try:
            if decoder_type == 'pymatching':
                import pymatching
                matcher = pymatching.Matching.from_detector_error_model(dem)
                predictions = matcher.decode_batch(detector_samples)
            else:
                raise ValueError(f"不支持的解码器类型: {decoder_type}")
            
            # 如果没有提供observables，假设全0（仅用于测试）
            if observables is None:
                observables = np.zeros(len(predictions), dtype=np.uint8)
            
            # 计算错误率
            errors = np.sum(predictions != observables)
            ler = errors / len(predictions)
            
            return ler
            
        except Exception as e:
            # 如果解码失败，返回一个惩罚值
            print(f"解码失败: {e}")
            return 1.0
    
    def get_detector_error_model(self, circuit: stim.Circuit = None) -> stim.DetectorErrorModel:
        """
        获取GRPO优化后的探测器错误模型
        
        Args:
            circuit: Stim电路对象（可选）
            
        Returns:
            优化后的DEM
        """
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self._build_dem_from_hyperedge_probs(self.hyperedge_probs)

