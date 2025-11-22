"""数据管理工具，用于保存和加载训练/评测数据"""

import pickle
import json
import os
from typing import Dict, Any, Optional
import numpy as np
import stim


class DataManager:
    """数据管理器，负责训练数据和结果的持久化"""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_training_data(self, 
                          experiment_name: str,
                          circuit: stim.Circuit,
                          detector_samples: np.ndarray,
                          observables: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        保存训练数据
        
        Args:
            experiment_name: 实验名称
            circuit: Stim电路
            detector_samples: 探测器采样数据
            observables: 可观测量数据
            metadata: 元数据（如code类型、distance、rounds等）
            
        Returns:
            保存的目录路径
        """
        exp_dir = os.path.join(self.base_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存电路
        circuit_path = os.path.join(exp_dir, "circuit.stim")
        with open(circuit_path, 'w') as f:
            f.write(str(circuit))
        
        # 保存ground truth DEM（用于评测时的真实值）
        dem_path = os.path.join(exp_dir, "ground_truth.dem")
        dem = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True
        )
        with open(dem_path, 'w') as f:
            f.write(str(dem))
        
        # 保存采样数据
        data_path = os.path.join(exp_dir, "samples.npz")
        np.savez_compressed(data_path, 
                          detectors=detector_samples,
                          observables=observables)
        
        # 保存元数据
        if metadata is None:
            metadata = {}
        metadata_path = os.path.join(exp_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return exp_dir
    
    def load_training_data(self, experiment_name: str) -> Dict[str, Any]:
        """
        加载训练数据
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            包含circuit, detector_samples, observables, metadata的字典
        """
        exp_dir = os.path.join(self.base_dir, experiment_name)
        
        # 加载电路
        circuit_path = os.path.join(exp_dir, "circuit.stim")
        with open(circuit_path, 'r') as f:
            circuit = stim.Circuit(f.read())
        
        # 加载采样数据
        data_path = os.path.join(exp_dir, "samples.npz")
        data = np.load(data_path)
        detector_samples = data['detectors']
        observables = data['observables']
        
        # 加载元数据
        metadata_path = os.path.join(exp_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'circuit': circuit,
            'detector_samples': detector_samples,
            'observables': observables,
            'metadata': metadata
        }
    
    def save_prediction_results(self,
                                experiment_name: str,
                                method_name: str,
                                hyperedge_probs: Dict,
                                additional_data: Optional[Dict] = None) -> str:
        """
        保存预测结果
        
        Args:
            experiment_name: 实验名称
            method_name: 预测方法名称
            hyperedge_probs: 超边概率字典
            additional_data: 其他需要保存的数据
            
        Returns:
            保存的文件路径
        """
        exp_dir = os.path.join(self.base_dir, experiment_name, "predictions")
        os.makedirs(exp_dir, exist_ok=True)
        
        result_path = os.path.join(exp_dir, f"{method_name}.pkl")
        
        save_data = {
            'hyperedge_probs': hyperedge_probs,
            'method_name': method_name
        }
        
        if additional_data:
            save_data.update(additional_data)
        
        with open(result_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        return result_path
    
    def load_prediction_results(self, experiment_name: str, method_name: str) -> Dict:
        """
        加载预测结果
        
        Args:
            experiment_name: 实验名称
            method_name: 预测方法名称
            
        Returns:
            预测结果字典
        """
        result_path = os.path.join(self.base_dir, experiment_name, "predictions", f"{method_name}.pkl")
        
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def save_evaluation_results(self,
                               experiment_name: str,
                               evaluator_name: str,
                               results: Dict) -> str:
        """
        保存评测结果
        
        Args:
            experiment_name: 实验名称
            evaluator_name: 评测器名称
            results: 评测结果
            
        Returns:
            保存的文件路径
        """
        exp_dir = os.path.join(self.base_dir, experiment_name, "evaluations")
        os.makedirs(exp_dir, exist_ok=True)
        
        result_path = os.path.join(exp_dir, f"{evaluator_name}.json")
        
        # 将结果转换为可序列化的格式
        serializable_results = self._make_serializable(results)
        
        with open(result_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return result_path
    
    @staticmethod
    def _make_serializable(obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {str(k): DataManager._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataManager._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def list_experiments(self) -> list:
        """列出所有实验"""
        if not os.path.exists(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir) 
                if os.path.isdir(os.path.join(self.base_dir, d))]

