"""超图权重预测方法的基类定义"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import stim
import numpy as np


class BasePredictor(ABC):
    """
    超图权重预测方法的基类
    
    所有预测方法都需要继承这个基类并实现其抽象方法
    """
    
    def __init__(self, name: str):
        """
        初始化预测器
        
        Args:
            name: 预测器名称
        """
        self.name = name
        self.trained = False
        self.hyperedge_probs = {}  # 存储预测的超边概率
    
    @abstractmethod
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        """
        训练预测器
        
        Args:
            circuit: Stim电路对象
            detector_samples: 探测器采样数据，形状为 (n_shots, n_detectors)
            **kwargs: 其他训练参数
            
        Returns:
            包含训练结果的字典，至少包含 'hyperedge_probs' 键
        """
        pass
    
    @abstractmethod
    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        """
        预测超边概率
        
        Args:
            circuit: Stim电路对象
            
        Returns:
            超边到概率的映射字典
        """
        pass
    
    def get_detector_error_model(self, circuit: stim.Circuit) -> stim.DetectorErrorModel:
        """
        获取基于预测概率的探测器错误模型
        
        Args:
            circuit: Stim电路对象
            
        Returns:
            更新后的探测器错误模型
        """
        if not self.trained:
            raise RuntimeError("预测器尚未训练，请先调用 train() 方法")
        
        return self._build_dem_from_probs(circuit, self.hyperedge_probs)
    
    @staticmethod
    def _build_dem_from_probs(circuit: stim.Circuit, hyperedge_probs: Dict[Tuple, float]) -> stim.DetectorErrorModel:
        """
        从超边概率构建DEM
        
        Args:
            circuit: Stim电路对象
            hyperedge_probs: 超边概率字典
            
        Returns:
            探测器错误模型
        """
        # 这个方法需要根据具体实现调整
        # 这里提供一个基本框架
        dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in hyperedge_probs.items():
            if prob > 0:
                targets = [stim.DemTarget(f"D{idx}") for idx in hyperedge]
                instruction = stim.DemInstruction("error", [prob], targets)
                dem.append(instruction)
        
        return dem
    
    def save_results(self, filepath: str) -> None:
        """
        保存训练结果
        
        Args:
            filepath: 保存路径
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'hyperedge_probs': self.hyperedge_probs,
                'trained': self.trained
            }, f)
    
    def load_results(self, filepath: str) -> None:
        """
        加载训练结果
        
        Args:
            filepath: 加载路径
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.hyperedge_probs = data['hyperedge_probs']
            self.trained = data['trained']

