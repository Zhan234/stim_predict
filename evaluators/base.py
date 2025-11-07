"""评测器基类定义"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import stim
import numpy as np


class BaseEvaluator(ABC):
    """
    评测器基类
    
    所有评测器都需要继承这个基类并实现其抽象方法
    """
    
    def __init__(self, name: str):
        """
        初始化评测器
        
        Args:
            name: 评测器名称
        """
        self.name = name
    
    @abstractmethod
    def evaluate(self,
                predicted_probs: Dict,
                ground_truth_probs: Dict,
                circuit: stim.Circuit,
                detector_samples: np.ndarray,
                observables: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        评测预测结果
        
        Args:
            predicted_probs: 预测的超边概率字典
            ground_truth_probs: 真实的超边概率字典
            circuit: Stim电路对象
            detector_samples: 探测器采样数据
            observables: 观测量真值
            **kwargs: 其他评测参数
            
        Returns:
            评测结果字典
        """
        pass
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """
        保存评测结果
        
        Args:
            results: 评测结果
            filepath: 保存路径
        """
        import json
        
        # 转换为可序列化格式
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    @staticmethod
    def _make_serializable(obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {str(k): BaseEvaluator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BaseEvaluator._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

