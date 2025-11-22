"""基于强化学习的超图权重预测方法"""

import stim
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
import correlation
from copy import deepcopy

from .base import BasePredictor


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
                 num_workers: int = 8):
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
            num_workers: 并行工作线程数
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
        self.num_workers = num_workers
        
        # 策略参数（高斯分布的均值和协方差）
        self.policy_mean = None
        self.policy_std = None
        
        # 旧策略参数（用于计算importance ratio，PPO算法核心）
        self.old_policy_mean = None
        self.old_policy_std = None
        
        # Baseline（对应论文中的ba，每个agent的baseline）
        self.baseline = None
        
        # Tanner图
        self.tanner_graph = None
        
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
        
        # 获取原始DEM和Tanner图
        dem_origin = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True
        )
        self.tanner_graph = correlation.TannerGraph(dem_origin)
        
        # 初始化策略参数
        n_params = len(self.tanner_graph.hyperedge_probs)
        
        # 从原始概率初始化（对数空间）
        init_probs = np.array([p for p in self.tanner_graph.hyperedge_probs.values()])
        init_probs = np.clip(init_probs, 1e-10, 1.0)  # 避免log(0)
        
        self.policy_mean = np.log(init_probs)  # 在对数空间中优化
        self.policy_std = np.ones(n_params) * 0.5  # 初始标准差
        
        # 初始化旧策略（用于PPO importance ratio）
        self.old_policy_mean = self.policy_mean.copy()
        self.old_policy_std = self.policy_std.copy()
        
        # 初始化baseline（论文方程G9，每个agent一个baseline，但简化为全局）
        # 在sensor场景下应该是每个sensor一个，这里简化为单一baseline
        self.baseline = 0.0
        
        print(f"开始RL训练，共 {self.epochs} 轮，批次大小 {self.batch_size}")
        
        # PPO训练循环
        for epoch in range(self.epochs):
            # 1. 采样参数候选
            log_probs_samples = self._sample_policy(self.batch_size)
            probs_samples = np.exp(log_probs_samples)  # 转回概率空间
            
            # 2. 评估每个候选的LER（reward）
            rewards = []
            lers = []
            
            for i in range(self.batch_size):
                hyperedge_probs_candidate = {}
                for j, hyperedge in enumerate(self.tanner_graph.hyperedge_probs.keys()):
                    hyperedge_probs_candidate[hyperedge] = probs_samples[i, j]
                
                # 构建候选DEM
                candidate_dem = self._build_dem_from_hyperedge_probs(hyperedge_probs_candidate)
                
                # 计算LER
                ler = self._evaluate_ler(
                    candidate_dem,
                    detector_samples,
                    observables,
                    decoder_type
                )
                
                # 计算reward: -log10(LER)
                reward = -np.log10(max(ler, 1e-10))
                
                rewards.append(reward)
                lers.append(ler)
            
            rewards = np.array(rewards)
            lers = np.array(lers)
            
            # 3. 计算优势函数（论文方程G10）
            advantages = rewards - self.baseline
            
            # 4. PPO策略更新（论文方程G23）
            policy_loss = self._update_policy_ppo(
                log_probs_samples,
                advantages
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
        final_probs = np.exp(self.policy_mean)
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
    
    def _sample_policy(self, n_samples: int) -> np.ndarray:
        """
        从当前策略采样参数
        
        Args:
            n_samples: 采样数量
            
        Returns:
            采样的对数概率数组，形状 (n_samples, n_params)
        """
        n_params = len(self.policy_mean)
        samples = np.random.normal(
            loc=self.policy_mean,
            scale=self.policy_std,
            size=(n_samples, n_params)
        )
        return samples
    
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
                          log_probs_samples: np.ndarray,
                          advantages: np.ndarray) -> float:
        """
        使用PPO更新策略（基于论文Appendix G）
        
        Args:
            log_probs_samples: 采样的对数概率
            advantages: 优势函数值
            
        Returns:
            策略损失
        """
        n_samples = len(advantages)
        
        # 计算importance ratio χ_θθ̃（论文方程G17和G20）
        # log π_θ(p) = -0.5 * ((p - μ) / σ)^2 - log(σ) - 0.5*log(2π)
        # log π_θ(p) - log π_θ̃(p) = -0.5*[(p-μ)/σ]^2 + 0.5*[(p-μ̃)/σ̃]^2 - log(σ/σ̃)
        
        log_ratios = []
        for i in range(n_samples):
            # 当前策略的对数概率
            z_new = (log_probs_samples[i] - self.policy_mean) / (self.policy_std + 1e-10)
            log_prob_new = -0.5 * np.sum(z_new ** 2) - np.sum(np.log(self.policy_std + 1e-10))
            
            # 旧策略的对数概率
            z_old = (log_probs_samples[i] - self.old_policy_mean) / (self.old_policy_std + 1e-10)
            log_prob_old = -0.5 * np.sum(z_old ** 2) - np.sum(np.log(self.old_policy_std + 1e-10))
            
            # Importance ratio
            log_ratio = log_prob_new - log_prob_old
            log_ratios.append(log_ratio)
        
        log_ratios = np.array(log_ratios)
        importance_ratios = np.exp(np.clip(log_ratios, -10, 10))  # 防止数值溢出
        
        # PPO clipping（论文Table I：repetition code用0.15，surface code用0.4）
        importance_ratios_clipped = np.clip(importance_ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        
        # 标准化advantages（减少方差）
        advantages_normalized = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 计算策略梯度（论文方程G21-G22）
        # ∇_θ J ≈ E[α(p) · χ_θθ̃(p)]
        gradients = np.zeros_like(self.policy_mean)
        
        for i in range(n_samples):
            # 使用clipped importance ratio
            weighted_advantage = advantages_normalized[i] * importance_ratios_clipped[i]
            
            # 高斯策略的梯度: ∇_μ log π(p|μ,σ) = (p - μ) / σ²
            grad_sample = weighted_advantage * (log_probs_samples[i] - self.policy_mean) / (self.policy_std ** 2 + 1e-10)
            gradients += grad_sample
        
        gradients /= n_samples
        
        # 梯度裁剪（论文Table I: 0.1）
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.max_grad_norm:
            gradients = gradients * self.max_grad_norm / grad_norm
        
        # 更新策略均值前，保存旧策略
        self.old_policy_mean = self.policy_mean.copy()
        self.old_policy_std = self.policy_std.copy()
        
        # 更新策略均值
        self.policy_mean += self.learning_rate * gradients
        
        # 更新标准差（逐渐减小探索）
        self.policy_std = np.maximum(self.policy_std * 0.995, 0.1)
        
        # 计算policy loss用于监控（论文方程G23）
        # Lpolicy = -E[α(p) · χ_θθ̃(p)]
        policy_loss = -np.mean(advantages_normalized * importance_ratios_clipped)
        
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

