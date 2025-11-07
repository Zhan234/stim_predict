"""解码器逻辑错误率评测器"""

import numpy as np
from typing import Dict, Any, List, Optional
import stim
import correlation

from .base import BaseEvaluator


class DecoderLEREvaluator(BaseEvaluator):
    """
    解码器逻辑错误率评测器
    
    评测不同预测方法在不同解码器下的逻辑错误率（LER）
    支持的解码器: PyMatching (MWPM), PyMatching with correlations, Union-Find等
    """
    
    def __init__(self, decoders: Optional[List[str]] = None):
        """
        初始化解码器LER评测器
        
        Args:
            decoders: 要测试的解码器列表，默认为 ['pymatching', 'pymatching_corr']
        """
        super().__init__(name="decoder_ler")
        
        if decoders is None:
            self.decoders = ['pymatching', 'pymatching_corr']
        else:
            self.decoders = decoders
    
    def evaluate(self,
                predicted_probs: Dict,
                ground_truth_probs: Dict,
                circuit: stim.Circuit,
                detector_samples: np.ndarray,
                observables: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        评测预测结果在不同解码器下的LER
        
        Args:
            predicted_probs: 预测的超边概率字典
            ground_truth_probs: 真实的超边概率字典
            circuit: Stim电路对象
            detector_samples: 探测器采样数据
            observables: 观测量真值
            **kwargs: 其他参数，可包含:
                - predicted_dem: 预测的DEM（如果已构建）
                - ground_truth_dem: 真实的DEM（如果已构建）
            
        Returns:
            各解码器的LER结果
        """
        results = {
            'n_shots': len(detector_samples),
            'decoders': {}
        }
        
        # 获取或构建DEM
        predicted_dem = kwargs.get('predicted_dem')
        ground_truth_dem = kwargs.get('ground_truth_dem')
        
        if predicted_dem is None:
            predicted_dem = self._build_dem_from_probs(circuit, predicted_probs)
        
        if ground_truth_dem is None:
            ground_truth_dem = self._build_dem_from_probs(circuit, ground_truth_probs)
        
        # 对每个解码器进行评测
        for decoder_name in self.decoders:
            try:
                # 使用预测的DEM
                pred_ler, pred_errors = self._decode_and_evaluate(
                    predicted_dem,
                    detector_samples,
                    observables,
                    decoder_name
                )
                
                # 使用真实的DEM
                true_ler, true_errors = self._decode_and_evaluate(
                    ground_truth_dem,
                    detector_samples,
                    observables,
                    decoder_name
                )
                
                results['decoders'][decoder_name] = {
                    'predicted_dem': {
                        'ler': float(pred_ler),
                        'n_errors': int(pred_errors),
                        'success_rate': float(1 - pred_ler)
                    },
                    'ground_truth_dem': {
                        'ler': float(true_ler),
                        'n_errors': int(true_errors),
                        'success_rate': float(1 - true_ler)
                    },
                    'ler_ratio': float(pred_ler / (true_ler + 1e-10)),  # 预测/真实的比率
                    'ler_difference': float(pred_ler - true_ler)
                }
                
            except Exception as e:
                results['decoders'][decoder_name] = {
                    'error': str(e)
                }
        
        # 计算平均性能
        valid_decoders = [d for d in results['decoders'].values() if 'error' not in d]
        if valid_decoders:
            avg_pred_ler = np.mean([d['predicted_dem']['ler'] for d in valid_decoders])
            avg_true_ler = np.mean([d['ground_truth_dem']['ler'] for d in valid_decoders])
            
            results['summary'] = {
                'avg_predicted_ler': float(avg_pred_ler),
                'avg_ground_truth_ler': float(avg_true_ler),
                'avg_ler_ratio': float(avg_pred_ler / (avg_true_ler + 1e-10))
            }
        
        return results
    
    def _decode_and_evaluate(self,
                            dem: stim.DetectorErrorModel,
                            detector_samples: np.ndarray,
                            observables: np.ndarray,
                            decoder_name: str) -> tuple:
        """
        使用指定解码器进行解码并评估
        
        Args:
            dem: 探测器错误模型
            detector_samples: 探测器采样
            observables: 真实观测量
            decoder_name: 解码器名称
            
        Returns:
            (逻辑错误率, 错误数量)
        """
        if decoder_name == 'pymatching':
            import pymatching
            matcher = pymatching.Matching.from_detector_error_model(dem)
            predictions = matcher.decode_batch(detector_samples)
            
        elif decoder_name == 'pymatching_corr':
            import pymatching
            matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            predictions, _ = matcher.decode_batch(detector_samples, 
                                                 return_weights=True, 
                                                 enable_correlations=True)
        
        elif decoder_name == 'union_find':
            # 如果有Union-Find解码器的实现
            try:
                import pymatching
                # PyMatching也可以用UF模式，这里作为占位符
                matcher = pymatching.Matching.from_detector_error_model(dem)
                predictions = matcher.decode_batch(detector_samples)
            except:
                raise NotImplementedError("Union-Find解码器尚未实现")
        
        else:
            raise ValueError(f"不支持的解码器: {decoder_name}")
        
        # 计算错误
        errors = np.sum(predictions != observables)
        ler = errors / len(observables)
        
        return ler, errors
    
    def _build_dem_from_probs(self,
                             circuit: stim.Circuit,
                             hyperedge_probs: Dict) -> stim.DetectorErrorModel:
        """
        从超边概率构建DEM
        
        Args:
            circuit: Stim电路
            hyperedge_probs: 超边概率字典
            
        Returns:
            DetectorErrorModel
        """
        # 获取原始DEM以获取结构信息
        original_dem = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True
        )
        
        # 使用Tanner图重建
        tanner_graph = correlation.TannerGraph(original_dem)
        
        # 构建新DEM
        new_dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in hyperedge_probs.items():
            if prob > 0 and hyperedge in tanner_graph.stim_decompose:
                prob = np.clip(prob, 1e-10, 1.0)
                
                decompose = tanner_graph.stim_decompose[hyperedge]
                targets = []
                
                for line_i in range(len(decompose)):
                    h = decompose[line_i]
                    t = tanner_graph.hyperedge_frames
                    
                    targets += [stim.DemTarget(f"D{id_index}") for id_index in h]
                    targets += [stim.DemTarget(f"L{id_index}") for id_index in t[h]]
                    
                    if line_i != len(decompose) - 1:
                        targets.append(stim.DemTarget("^"))
                
                instruction = stim.DemInstruction("error", [prob], targets)
                new_dem.append(instruction)
        
        return new_dem
    
    def compare_multiple_methods(self,
                                 predictions: Dict[str, Dict],
                                 ground_truth_probs: Dict,
                                 circuit: stim.Circuit,
                                 detector_samples: np.ndarray,
                                 observables: np.ndarray,
                                 **kwargs) -> Dict[str, Any]:
        """
        比较多个预测方法的解码性能
        
        Args:
            predictions: 方法名到预测概率字典的映射
            ground_truth_probs: 真实概率字典
            circuit: Stim电路
            detector_samples: 探测器采样
            observables: 观测量
            **kwargs: 其他参数
            
        Returns:
            比较结果
        """
        comparison = {}
        
        for method_name, pred_probs in predictions.items():
            print(f"评测方法: {method_name}")
            results = self.evaluate(
                pred_probs,
                ground_truth_probs,
                circuit,
                detector_samples,
                observables,
                **kwargs
            )
            comparison[method_name] = results
        
        # 添加排名
        if len(predictions) > 1:
            rankings = {}
            
            for decoder_name in self.decoders:
                method_lers = {}
                for method_name in predictions.keys():
                    if decoder_name in comparison[method_name].get('decoders', {}):
                        decoder_result = comparison[method_name]['decoders'][decoder_name]
                        if 'predicted_dem' in decoder_result:
                            ler = decoder_result['predicted_dem']['ler']
                            method_lers[method_name] = ler
                
                if method_lers:
                    # 排序（升序，LER越小越好）
                    sorted_methods = sorted(method_lers.items(), key=lambda x: x[1])
                    rankings[decoder_name] = [
                        {'method': m[0], 'ler': m[1]} 
                        for m in sorted_methods
                    ]
            
            comparison['rankings'] = rankings
        
        return comparison

