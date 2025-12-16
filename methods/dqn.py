"""基于 DQN (DDPG) 的超图权重预测方法"""

import stim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Any, List, Optional
import correlation
from collections import deque
import random

from .base import BasePredictor


class QNetwork(nn.Module):
    """
    Critic Network (Q-Function)
    输入: 权重向量 (Action)
    输出: 预测的 Reward (-log LER)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class DQNPredictor(BasePredictor):
    """
    基于 DDPG (Deep Deterministic Policy Gradient) 的预测器
    
    由于超图权重是连续值，标准的 DQN (离散动作空间) 并不适用。
    这里我们实现 DDPG，它是 DQN 在连续动作空间上的扩展。
    
    核心思想:
    1. Actor: 维护一组确定性的权重参数 (Action)。
    2. Critic: 学习一个 Q 网络，预测给定权重的效果 (Reward)。
    3. Replay Buffer: 复用历史数据，提高样本效率。
    """
    
    def __init__(self,
                 learning_rate_actor: float = 1e-3,
                 learning_rate_critic: float = 1e-3,
                 buffer_size: int = 1000,
                 batch_size: int = 32,
                 epochs: int = 50,
                 exploration_noise: float = 2.0,  # 再次大幅增大初始噪声
                 warmup_epochs: int = 5,          # 减少预热
                 use_gpu: bool = True):
        super().__init__(name="dqn")
        
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.exploration_noise = exploration_noise
        self.warmup_epochs = warmup_epochs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"DQN(DDPG) 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("DQN(DDPG) 使用 CPU")
            
        # 模型组件
        self.actor_params = None  # 这是一个可学习的向量，即当前的最佳权重
        self.critic = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        
        # 经验回放池
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.tanner_graph = None
        self.hyperedge_probs = None
        
        self.training_history = {
            'rewards': [],
            'ler': [],
            'critic_loss': []
        }

    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        observables = kwargs.get('observables', None)
        decoder_type = kwargs.get('decoder_type', 'pymatching')
        
        # 1. 初始化
        # 获取原始DEM和Tanner图
        dem_origin = circuit.detector_error_model(decompose_errors=True)
        self.tanner_graph = correlation.TannerGraph(dem_origin)
        n_params = len(self.tanner_graph.hyperedge_probs)
        
        # 初始化 Actor (权重参数)
        # 从原始概率的对数开始
        init_probs = np.array([p for p in self.tanner_graph.hyperedge_probs.values()])
        init_probs = np.clip(init_probs, 1e-10, 1.0)
        # 我们优化的是 log(prob)，这样可以保证 prob > 0
        self.actor_params = nn.Parameter(
            torch.tensor(np.log(init_probs), dtype=torch.float32, device=self.device)
        )
        
        # 初始化 Critic
        self.critic = QNetwork(n_params).to(self.device)
        
        # 优化器
        self.optimizer_actor = optim.Adam([self.actor_params], lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        print(f"开始 DQN(DDPG) 训练，共 {self.epochs} 轮")
        
        for epoch in range(self.epochs):
            # --- 1. 探索与收集数据 (Interaction) ---
            
            # 获取当前动作 (Weights) 并添加噪声
            with torch.no_grad():
                current_action = self.actor_params.data.clone()
                
                # 噪声衰减：衰减到 10% 而不是 0，保持后期探索能力
                current_noise_scale = self.exploration_noise * (1.0 - 0.9 * epoch / self.epochs)
                noise = torch.randn_like(current_action) * current_noise_scale
                
                exploratory_action = current_action + noise
                
                # 转换回概率空间用于评估
                probs_np = torch.exp(exploratory_action).cpu().numpy()
            
            # 构建 DEM 并评估
            hyperedge_probs_map = {}
            for i, key in enumerate(self.tanner_graph.hyperedge_probs.keys()):
                hyperedge_probs_map[key] = probs_np[i]
            
            candidate_dem = self._build_dem_from_hyperedge_probs(hyperedge_probs_map)
            ler = self._evaluate_ler(candidate_dem, detector_samples, observables, decoder_type)
            reward = -np.log10(max(ler, 1e-10))
            
            # 存入 Buffer: (Action, Reward)
            self.replay_buffer.append((exploratory_action.cpu().numpy(), reward))
            
            # 记录
            self.training_history['rewards'].append(reward)
            self.training_history['ler'].append(ler)
            
            # --- 2. 训练 Critic (Policy Evaluation) ---
            critic_loss_val = 0.0
            # 只有当 Buffer 足够大时才开始训练
            if len(self.replay_buffer) > self.batch_size:
                # 多次更新 Critic 以提高准确性
                for _ in range(5):
                    batch = random.sample(self.replay_buffer, self.batch_size)
                    b_actions = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32, device=self.device)
                    b_rewards = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
                    
                    # Critic Loss: MSE(Q(a), r)
                    q_values = self.critic(b_actions)
                    critic_loss = nn.MSELoss()(q_values, b_rewards)
                    
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    self.optimizer_critic.step()
                    critic_loss_val = critic_loss.item()
            
            self.training_history['critic_loss'].append(critic_loss_val)
            
            # --- 3. 训练 Actor (Policy Improvement) ---
            # 预热期不更新 Actor，先让 Critic 学会评价
            if epoch >= self.warmup_epochs and len(self.replay_buffer) > self.batch_size:
                q_val = self.critic(self.actor_params.unsqueeze(0)) # Add batch dim
                actor_loss = -q_val.mean()
                
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"Reward={reward:.3f}, LER={ler:.6f}, "
                      f"Critic Loss={critic_loss_val:.4f}, "
                      f"Noise={current_noise_scale:.2f}")
        
        # 训练结束，保存结果
        final_probs = torch.exp(self.actor_params).detach().cpu().numpy()
        self.hyperedge_probs = {}
        for i, key in enumerate(self.tanner_graph.hyperedge_probs.keys()):
            self.hyperedge_probs[key] = final_probs[i]
            
        self.trained = True
        return {
            'hyperedge_probs': self.hyperedge_probs,
            'training_history': self.training_history
        }

    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        return self.hyperedge_probs

    def _build_dem_from_hyperedge_probs(self, hyperedge_probs: Dict) -> stim.DetectorErrorModel:
        """
        从超边概率构建DEM (复用 RLBasedPredictor 的逻辑)
        """
        dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in hyperedge_probs.items():
            if prob > 0:
                prob = np.clip(prob, 1e-10, 1.0)
                
                # 注意：这里依赖 correlation 库的 TannerGraph 实现
                if hasattr(self.tanner_graph, 'stim_decompose'):
                    decompose = self.tanner_graph.stim_decompose[hyperedge]
                    targets = []
                    
                    for line_i in range(len(decompose)):
                        h = decompose[line_i]
                        # hyperedge_frames 似乎是逻辑观测量的映射
                        t = getattr(self.tanner_graph, 'hyperedge_frames', {})
                        
                        targets += [stim.DemTarget(f"D{id_index}") for id_index in h]
                        if h in t:
                            targets += [stim.DemTarget(f"L{id_index}") for id_index in t[h]]
                        
                        if line_i != len(decompose) - 1:
                            targets.append(stim.DemTarget("^"))
                    
                    instruction = stim.DemInstruction("error", [prob], targets)
                    dem.append(instruction)
                else:
                    # 备用逻辑：如果无法分解，尝试直接构建
                    targets = [stim.DemTarget(f"D{id_index}") for id_index in hyperedge]
                    dem.append(stim.DemInstruction("error", [prob], targets))
        
        return dem

    def _evaluate_ler(self, dem, samples, observables, decoder_type):
        """
        评估给定DEM的逻辑错误率
        """
        try:
            if decoder_type == 'pymatching':
                import pymatching
                matcher = pymatching.Matching.from_detector_error_model(dem)
                predictions = matcher.decode_batch(samples)
            else:
                raise ValueError(f"不支持的解码器类型: {decoder_type}")
            
            if observables is None:
                observables = np.zeros(len(predictions), dtype=np.uint8)
            
            errors = np.sum(predictions != observables)
            ler = errors / len(predictions)
            
            return ler
            
        except Exception as e:
            print(f"解码失败: {e}")
            return 1.0
