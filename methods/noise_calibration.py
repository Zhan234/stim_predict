"""基于噪声校准的超图权重预测方法"""

import stim
import numpy as np
from typing import Dict, Tuple, Any
from copy import deepcopy
import correlation

from .base import BasePredictor


class NoiseCalibrationPredictor(BasePredictor):
    """
    噪声校准预测器
    
    基于实际探测器采样数据，通过计算高阶相关性来校准超边概率
    参考: dem.ipynb中的calibration函数
    """
    
    def __init__(self, num_workers: int = 8):
        super().__init__(name="noise_calibration")
        self.num_workers = num_workers
        self.dem_calibrated = None
        self.tanner_graph = None
    
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        # 获取原始DEM
        dem_origin = circuit.detector_error_model(
            decompose_errors=True, 
            approximate_disjoint_errors=True
        )
        
        # 构建Tanner图
        self.tanner_graph = correlation.TannerGraph(dem_origin)
        
        # 计算高阶相关性
        correlation_result = correlation.cal_high_order_correlations(
            detector_samples, 
            self.tanner_graph.hyperedges, 
            num_workers=self.num_workers
        )
        
        # 收集从相关性计算得到的概率
        prob_from_correlation = []
        for hyperedge in self.tanner_graph.hyperedge_probs.keys():
            prob = correlation_result.get(hyperedge)
            prob_from_correlation.append(prob)
        
        # 获取原始模型的最小概率（用于替换负概率）
        model = dem_origin.flattened()
        model_str = str(model)
        model_list = self._convert_model_str_2_model_list(model_str)
        model_list_del_detline = self._del_detector_line(model_list)
        prob_list = [float(line[0][6:-1]) for line in model_list_del_detline if line and line[0]]
        min_prob_model = min(prob_list) if prob_list else 1e-10
        
        # 处理负概率
        negative_prob_indices = []
        for prob_idx in range(len(prob_from_correlation)):
            if prob_from_correlation[prob_idx] is None or prob_from_correlation[prob_idx] <= 0:
                prob_from_correlation[prob_idx] = min_prob_model
                negative_prob_indices.append(prob_idx)
        
        # 构建校准后的DEM
        correlation_dem = stim.DetectorErrorModel()
        hyperedge_probs = {}
        
        for hyperedge, original_prob in self.tanner_graph.hyperedge_probs.items():
            p = correlation_result.get(hyperedge)
            
            # 如果相关性计算的概率无效，使用原始概率
            if p is None or p <= 0:
                p = original_prob
            
            hyperedge_probs[hyperedge] = p
            
            # 获取分解和目标
            decompose = self.tanner_graph.stim_decompose[hyperedge]
            targets = []
            
            for line_i in range(len(decompose)):
                h = decompose[line_i]
                t = self.tanner_graph.hyperedge_frames
                
                # 添加探测器目标
                targets += [stim.DemTarget(f"D{id_index}") for id_index in h]
                # 添加逻辑观测量目标
                targets += [stim.DemTarget(f"L{id_index}") for id_index in t[h]]
                
                # 添加分隔符（除了最后一行）
                if line_i != len(decompose) - 1:
                    targets.append(stim.DemTarget("^"))
            
            # 创建错误指令
            instruction = stim.DemInstruction("error", [p], targets)
            correlation_dem.append(instruction)
        
        self.dem_calibrated = correlation_dem
        self.hyperedge_probs = hyperedge_probs
        self.trained = True
        
        return {
            'hyperedge_probs': hyperedge_probs,
            'dem': correlation_dem,
            'negative_prob_indices': negative_prob_indices
        }
    
    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self.hyperedge_probs
    
    def get_detector_error_model(self, circuit: stim.Circuit = None) -> stim.DetectorErrorModel:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self.dem_calibrated
    
    @staticmethod
    def _convert_model_str_2_model_list(model_string: str) -> list:
        """
        将模型字符串转换为列表格式
        
        Args:
            model_string: DEM的字符串表示
            
        Returns:
            模型列表
        """
        model_list = list(model_string.split('\n'))
        for index_model_list in range(len(model_list)):
            model_list[index_model_list] = model_list[index_model_list].replace(', ', ',')
            model_list[index_model_list] = model_list[index_model_list].split(' ')
        return model_list
    
    @staticmethod
    def _del_detector_line(model_list: list) -> list:
        """
        删除模型列表中的探测器定义行
        
        Args:
            model_list: 模型列表
            
        Returns:
            删除探测器行后的模型列表
        """
        model_list_del_detline = deepcopy(model_list)
        for line_indx in range(len(model_list) - 1, -1, -1):
            if model_list_del_detline[line_indx] and model_list_del_detline[line_indx][0]:
                if model_list_del_detline[line_indx][0] and model_list_del_detline[line_indx][0][0] == 'd':
                    del model_list_del_detline[line_indx]
        return model_list_del_detline

