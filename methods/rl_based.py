"""基于强化学习的超图权重预测方法"""

import stim
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Optional
import correlation
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import os

from .base import BasePredictor
from .correlation import CorrelationPredictor


class RLBasedPredictor(BasePredictor):
    """
    基于强化学习的预测器
    
    使用PPO (Proximal Policy Optimization) 算法优化decoder priors
    参考论文: Sivak et al. 2024 - Optimization of decoder priors for accurate quantum error correction
    
    核心思想:
    1. 每个agent对应一个小的sensor code
    2. Agent的action是调整error hypergraph的参数
    3. Reward定义为 -log10(LER)
    4. 使用PPO更新策略
    """
    
    def __init__(self,
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 100,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_gpu: bool = True,
                 num_workers: int = 8,
                 eval_subset_size: int = 10000,
                 correlation_use_numerical: bool = True,
                 correlation_num_workers: int = 16):
        """
        初始化RL预测器

        Args:
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            clip_ratio: PPO重要性比率裁剪阈值
            entropy_coef: 熵损失系数（鼓励探索）
            value_coef: 价值函数损失系数
            max_grad_norm: 梯度裁剪的最大范数
            use_gpu: 是否使用GPU
            num_workers: 并行解码的工作线程数
            eval_subset_size: 每次评估使用的样本子集大小（0表示使用全部）
            correlation_use_numerical: Correlation方法是否使用数值方法
            correlation_num_workers: Correlation方法的并行线程数
        """
        super().__init__(name="rl_based")
        
        # 超参数
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_workers = num_workers
        self.eval_subset_size = eval_subset_size
        self.correlation_use_numerical = correlation_use_numerical
        self.correlation_num_workers = correlation_num_workers
        
        # 设备设置
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU")
        
        # 策略参数（使用PyTorch张量）
        self.policy_mean = None
        self.policy_std = None

        # 旧策略参数（用于计算importance ratio，PPO算法核心）
        self.old_policy_mean = None
        self.old_policy_std = None

        # Baseline（对应论文中的ba，每个agent的baseline）
        self.baseline = None

        # Tanner图
        self.tanner_graph = None

        # Correlation基线
        self.correlation_predictor = None
        self.baseline_probs = None
        
        # 训练历史
        self.training_history = {
            'rewards': [],
            'ler': [],
            'policy_loss': [],
            'value_loss': []
        }
    
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        """
        使用PPO训练RL预测器
        
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
        experiment_name = kwargs.get('experiment_name', None)

        print("=" * 80)
        print("步骤 1/2: 使用Correlation方法获取基线概率")
        print("=" * 80)

        # 获取correlation结果作为初始化
        correlation_loaded = False
        if experiment_name:
            try:
                # 尝试导入DataManager
                from ..utils.data_manager import DataManager
            except ImportError:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from utils.data_manager import DataManager

            data_manager = DataManager()
            corr_pred_path = os.path.join(data_manager.base_dir, experiment_name, "predictions", "correlation.pkl")
            if os.path.exists(corr_pred_path):
                print(f"发现已有的correlation结果，正在加载: {corr_pred_path}")
                try:
                    corr_data = data_manager.load_prediction_results(experiment_name, "correlation")
                    self.baseline_probs = corr_data['hyperedge_probs']
                    if 'tanner_graph' in corr_data:
                        self.tanner_graph = corr_data['tanner_graph']
                        print("成功加载tanner_graph")
                        if not hasattr(self.tanner_graph, 'hyperedge_probs') or not self.tanner_graph.hyperedge_probs:
                            print("警告: 加载的tanner_graph不完整，重新创建...")
                            dem_origin = circuit.detector_error_model(
                                decompose_errors=True,
                                approximate_disjoint_errors=True
                            )
                            self.tanner_graph = correlation.TannerGraph(dem_origin)
                    else:
                        print("警告: correlation结果中未包含tanner_graph，正在重新创建...")
                        dem_origin = circuit.detector_error_model(
                            decompose_errors=True,
                            approximate_disjoint_errors=True
                        )
                        self.tanner_graph = correlation.TannerGraph(dem_origin)

                    correlation_loaded = True
                    print(f"成功加载correlation结果，共 {len(self.baseline_probs)} 个超边")
                except Exception as e:
                    print(f"加载correlation结果失败: {e}，将重新训练")
                    correlation_loaded = False
            else:
                print(f"未找到correlation结果文件: {corr_pred_path}")

        if not correlation_loaded:
            print("未找到correlation结果，开始训练correlation方法...")
            # 使用Correlation方法获取基线概率
            self.correlation_predictor = CorrelationPredictor(
                use_numerical=self.correlation_use_numerical,
                num_workers=self.correlation_num_workers
            )

            corr_result = self.correlation_predictor.train(circuit, detector_samples)
            self.baseline_probs = corr_result['hyperedge_probs']
            self.tanner_graph = corr_result['tanner_graph']

        print(f"Correlation完成，{len(self.baseline_probs)} 个超边")

        # 初始化策略参数
        n_params = len(self.tanner_graph.hyperedge_probs)

        # 使用correlation结果作为初始化，避免平坦初始化
        init_probs = np.array([self.baseline_probs[hyperedge] for hyperedge in self.tanner_graph.hyperedge_probs.keys()])
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
        
        # 初始化旧策略（用于PPO importance ratio）
        self.old_policy_mean = self.policy_mean.clone().detach()
        self.old_policy_std = self.policy_std.clone().detach()
        
        # 初始化baseline（论文方程G9，每个agent一个baseline，但简化为全局）
        # 在sensor场景下应该是每个sensor一个，这里简化为单一baseline
        self.baseline = 0.0
        
        # 优化器
        optimizer = torch.optim.Adam(
            [self.policy_mean, self.policy_std],
            lr=self.learning_rate
        )
        
        print("\n" + "=" * 80)
        print("步骤 2/2: RL策略优化")
        print("=" * 80)

        print(f"开始RL训练，共 {self.epochs} 轮，批次大小 {self.batch_size}")
        print(f"并行工作线程: {self.num_workers}, 评估样本子集大小: {self.eval_subset_size if self.eval_subset_size > 0 else '全部'}")
        print(f"初始化概率范围: [{init_probs.min():.2e}, {init_probs.max():.2e}]")
        
        # PPO训练循环
        for epoch in range(self.epochs):
            # 1. 采样参数候选
            log_probs_samples, probs_samples = self._sample_policy(self.batch_size)
            probs_samples_np = probs_samples.cpu().numpy()  # 转回numpy用于后续处理
            
            # 2. 并行评估每个候选的LER（reward）
            rewards, lers = self._evaluate_batch_parallel(
                probs_samples_np,
                detector_samples,
                observables,
                decoder_type
            )
            
            rewards = np.array(rewards)
            lers = np.array(lers)
            
            # 3. 计算优势函数（论文方程G10）
            advantages = rewards - self.baseline
            
            # 4. PPO策略更新（论文方程G23）
            policy_loss = self._update_policy_ppo(
                log_probs_samples,
                advantages,
                optimizer
            )
            
            # 5. 更新baseline（论文方程G24）
            baseline_loss = self._update_baseline(advantages)
            
            # 6. 更新baseline的值（使用指数移动平均）
            self.baseline = 0.9 * self.baseline + 0.1 * np.mean(rewards)
            
            # 记录训练历史
            self.training_history['rewards'].append(np.mean(rewards))
            self.training_history['ler'].append(np.mean(lers))
            self.training_history['policy_loss'].append(policy_loss)
            self.training_history['value_loss'].append(baseline_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"Mean Reward={np.mean(rewards):.3f}, "
                      f"Mean LER={np.mean(lers):.6f}, "
                      f"Policy Loss={policy_loss:.4f}")
        
        # 使用最终策略的均值作为预测
        final_probs = torch.exp(self.policy_mean).detach().cpu().numpy()
        final_hyperedge_probs = {}
        for i, hyperedge in enumerate(self.tanner_graph.hyperedge_probs.keys()):
            final_hyperedge_probs[hyperedge] = final_probs[i]
        
        self.hyperedge_probs = final_hyperedge_probs
        self.trained = True
        
        return {
            'hyperedge_probs': final_hyperedge_probs,
            'baseline_probs': self.baseline_probs,
            'training_history': self.training_history,
            'final_mean_reward': self.training_history['rewards'][-1],
            'final_mean_ler': self.training_history['ler'][-1],
            'tanner_graph': self.tanner_graph
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
    
    def _sample_policy(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从当前策略采样参数
        
        Args:
            n_samples: 采样数量
            
        Returns:
            (log_probs_samples, probs_samples): 对数概率和概率张量，形状 (n_samples, n_params)
        """
        n_params = len(self.policy_mean)
        
        # 从高斯策略采样（在对数空间）
        with torch.no_grad():
            noise = torch.randn(
                (n_samples, n_params),
                device=self.device,
                dtype=torch.float32
            )
            log_probs_samples = self.policy_mean.unsqueeze(0) + noise * self.policy_std.unsqueeze(0)
        
        # 转回概率空间（并裁剪到有效范围）
        probs_samples = torch.clamp(torch.exp(log_probs_samples), min=1e-10, max=1.0)
        
        return log_probs_samples, probs_samples
    
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
    
    def _update_policy_ppo(self,
                          log_probs_samples: torch.Tensor,
                          advantages: np.ndarray,
                          optimizer: torch.optim.Optimizer) -> float:
        """
        使用PPO更新策略（基于论文Appendix G）
        
        Args:
            log_probs_samples: 采样的对数概率，形状 (batch_size, n_params)
            advantages: 优势函数值，形状 (batch_size,)
            optimizer: PyTorch优化器
            
        Returns:
            策略损失
        """
        batch_size, n_params = log_probs_samples.shape
        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )
        
        # 计算当前策略的对数概率
        # log π_θ(p) = -0.5 * ((p - μ) / σ)^2 - log(σ) - 0.5*log(2π)
        z_current = (log_probs_samples - self.policy_mean.unsqueeze(0)) / (self.policy_std.unsqueeze(0) + 1e-10)
        log_prob_current = -0.5 * torch.sum(z_current ** 2, dim=1) - torch.sum(torch.log(self.policy_std + 1e-10))
            
        # 计算旧策略的对数概率
        z_old = (log_probs_samples - self.old_policy_mean.unsqueeze(0)) / (self.old_policy_std.unsqueeze(0) + 1e-10)
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
        
        # 标准化advantages（减少方差）
        advantages_normalized = (advantages_tensor - torch.mean(advantages_tensor)) / (torch.std(advantages_tensor) + 1e-8)
        
        # PPO目标函数（最大化，所以取负号作为损失）
        # Lpolicy = -E[min(ratio * A, clipped_ratio * A)]
        policy_objective = torch.min(
            importance_ratio * advantages_normalized,
            importance_ratio_clipped * advantages_normalized
        )
        loss = -torch.mean(policy_objective)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [self.policy_mean, self.policy_std],
            self.max_grad_norm
        )
        
        optimizer.step()
        
        # 更新旧策略（用于下一轮的importance ratio）
        self.old_policy_mean = self.policy_mean.clone().detach()
        self.old_policy_std = self.policy_std.clone().detach()
        
        # 更新标准差（逐渐减小探索）
        with torch.no_grad():
            self.policy_std.data = torch.clamp(self.policy_std.data * 0.995, min=0.1)
        
        # 计算policy loss用于监控
        policy_loss = -torch.mean(policy_objective).item()
        
        return policy_loss
    
    def _update_baseline(self, advantages: np.ndarray) -> float:
        """
        更新baseline（论文方程G24）
        
        Args:
            advantages: 优势函数值
            
        Returns:
            baseline损失
        """
        # 论文方程G24: Lbaseline = E[||α(p)||²]
        # 这个损失用于监控baseline的质量
        # baseline越好，advantages的方差越小
        
        baseline_loss = np.mean(advantages ** 2)
        
        return baseline_loss
    
    def get_detector_error_model(self, circuit: stim.Circuit = None) -> stim.DetectorErrorModel:
        """
        获取RL优化后的探测器错误模型
        
        Args:
            circuit: Stim电路对象（可选）
            
        Returns:
            优化后的DEM
        """
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self._build_dem_from_hyperedge_probs(self.hyperedge_probs)

