"""电路生成工厂，支持多种量子纠错码"""

import stim
from typing import Dict, Any, Optional
import numpy as np


class CircuitFactory:
    """
    支持生成不同类型的量子纠错码电路，包括：
    - Surface Code (表面码)
    - Repetition Code (重复码)
    - Color Code (颜色码)
    等
    """
    
    # 噪声参数类型定义
    ID = 'ID'  # 空闲噪声
    CX = 'CX'  # 两比特门噪声
    PX = 'PX'  # X基制备噪声
    PZ = 'PZ'  # Z基制备噪声
    MX = 'MX'  # X基测量噪声
    MZ = 'MZ'  # Z基测量噪声
    
    @staticmethod
    def create_circuit(code_type: str,
                      distance: int,
                      rounds: int,
                      noise_model: Optional[Dict[str, float]] = None,
                      **kwargs) -> stim.Circuit:
        """
        创建量子纠错码电路
        
        Args:
            code_type: 编码类型，如 'surface_code', 'repetition_code' 等
            distance: 码距
            rounds: 测量轮数
            noise_model: 噪声模型参数字典
            **kwargs: 其他参数
            
        Returns:
            Stim电路对象
        """
        if code_type == 'surface_code':
            return CircuitFactory._create_surface_code(distance, rounds, noise_model, **kwargs)
        elif code_type == 'repetition_code':
            return CircuitFactory._create_repetition_code(distance, rounds, noise_model, **kwargs)
        elif code_type == 'color_code':
            return CircuitFactory._create_color_code(distance, rounds, noise_model, **kwargs)
        else:
            raise ValueError(f"不支持的编码类型: {code_type}")
    
    @staticmethod
    def _create_surface_code(distance: int,
                            rounds: int,
                            noise_model: Optional[Dict[str, float]] = None,
                            memory_type: str = 'z',
                            **kwargs) -> stim.Circuit:
        """
        创建表面码电路
        
        Args:
            distance: 码距
            rounds: 测量轮数
            noise_model: 噪声模型
            memory_type: 内存类型，'z' 或 'x'
            
        Returns:
            表面码电路
        """
        # 默认噪声参数
        if noise_model is None:
            p = kwargs.get('noise_level', 0.001)
            noise_model = {
                CircuitFactory.ID: p,
                CircuitFactory.CX: p,
                CircuitFactory.PX: p,
                CircuitFactory.PZ: p,
                CircuitFactory.MX: p,
                CircuitFactory.MZ: p,
            }
        
        # 使用stim内置生成器生成基础电路
        task = f'surface_code:rotated_memory_{memory_type}'
        circuit = stim.Circuit.generated(
            task,
            distance=distance,
            rounds=rounds
        )
        
        # 修正OBSERVABLE_INCLUDE
        # 原始电路最后一条指令可能不正确，需要修正
        if circuit[-1].name == 'OBSERVABLE_INCLUDE':
            circuit = circuit[:-1]
            # 重新添加正确的OBSERVABLE_INCLUDE
            circuit.append(
                stim.CircuitInstruction('OBSERVABLE_INCLUDE', 
                                       [stim.target_rec(-j - 1) for j in range(distance)], 
                                       [0])
            )
        
        # 添加噪声
        circuit_with_noise = CircuitFactory._add_noise_to_circuit(circuit, noise_model)
        
        return circuit_with_noise
    
    @staticmethod
    def _create_repetition_code(distance: int,
                               rounds: int,
                               noise_model: Optional[Dict[str, float]] = None,
                               **kwargs) -> stim.Circuit:
        """
        创建重复码电路
        
        Args:
            distance: 码距
            rounds: 测量轮数
            noise_model: 噪声模型
            
        Returns:
            重复码电路
        """
        # 默认噪声参数
        if noise_model is None:
            p = kwargs.get('noise_level', 0.01)
            noise_model = {
                'after_clifford_depolarization': p,
                'before_round_data_depolarization': p,
                'before_measure_flip_probability': p,
                'after_reset_flip_probability': p,
            }
        
        # 使用stim内置生成器
        circuit = stim.Circuit.generated(
            code_task='repetition_code:memory',
            distance=distance,
            rounds=rounds,
            **noise_model
        )
        
        return circuit
    
    @staticmethod
    def _create_color_code(distance: int,
                          rounds: int,
                          noise_model: Optional[Dict[str, float]] = None,
                          **kwargs) -> stim.Circuit:
        """
        创建颜色码电路
        
        Args:
            distance: 码距
            rounds: 测量轮数
            noise_model: 噪声模型
            
        Returns:
            颜色码电路
        """
        # 默认噪声参数
        if noise_model is None:
            p = kwargs.get('noise_level', 0.001)
            noise_model = {
                'after_clifford_depolarization': p,
                'after_reset_flip_probability': p,
                'before_measure_flip_probability': p,
                'before_round_data_depolarization': p,
            }
        
        # 使用stim内置生成器
        circuit = stim.Circuit.generated(
            code_task='color_code:memory_xyz',
            distance=distance,
            rounds=rounds,
            **noise_model
        )
        
        return circuit
    
    @staticmethod
    def _add_noise_to_circuit(circuit: stim.Circuit,
                             noise_params: Dict[str, float],
                             qubits: Optional[set] = None,
                             occupied_qubits: Optional[set] = None) -> stim.Circuit:
        """
        向电路添加噪声（基于dem.ipynb中的add_noise函数）
        
        Args:
            circuit: 原始电路
            noise_params: 噪声参数字典
            qubits: 比特集合
            occupied_qubits: 已占用比特集合
            
        Returns:
            添加噪声后的电路
        """
        MZ = CircuitFactory.MZ
        MX = CircuitFactory.MX
        PZ = CircuitFactory.PZ
        PX = CircuitFactory.PX
        ID = CircuitFactory.ID
        CX = CircuitFactory.CX
        
        # 检查对称性假设
        assert noise_params.get(MZ, 0) == noise_params.get(MX, 0)
        assert noise_params.get(PZ, 0) == noise_params.get(PX, 0)
        
        circuit_new = stim.Circuit()
        if qubits is None:
            qubits = circuit.get_final_qubit_coordinates().keys()
        if occupied_qubits is None:
            occupied_qubits = set()
        
        for inst in circuit:
            if inst.name == 'REPEAT':
                # 递归处理REPEAT块
                circuit_new.append(
                    stim.CircuitRepeatBlock(
                        inst.repeat_count,
                        CircuitFactory._add_noise_to_circuit(
                            inst.body_copy(), noise_params, qubits, occupied_qubits
                        )
                    )
                )
            else:
                targets = inst.targets_copy()
                
                # 标记占用的比特
                if inst.name in ['R', 'M', 'MR', 'CX']:
                    occupied_qubits.update(target.qubit_value for target in targets)
                    
                    if inst.name in ['M', 'MR']:
                        # 添加测量噪声
                        circuit_new.append(
                            stim.CircuitInstruction('X_ERROR', targets, [noise_params[MZ]])
                        )
                        if inst.name == 'MR':
                            occupied_qubits.add(-1)  # 标记双长度tick
                
                elif inst.name == 'TICK':
                    # 添加空闲噪声
                    if occupied_qubits:
                        idle_targets = [stim.GateTarget(i) for i in qubits if i not in occupied_qubits]
                        if idle_targets:
                            if -1 in occupied_qubits:  # MR在这个tick中发生
                                idle_targets += idle_targets
                            circuit_new.append(
                                stim.CircuitInstruction('DEPOLARIZE1', idle_targets, [noise_params[ID]])
                            )
                    occupied_qubits = set()
                
                # 添加原始指令
                circuit_new.append(inst)
                
                # 添加门后噪声
                if inst.name in ['R', 'MR']:
                    circuit_new.append(
                        stim.CircuitInstruction('X_ERROR', targets, [noise_params[PZ]])
                    )
                elif inst.name == 'CX':
                    circuit_new.append(
                        stim.CircuitInstruction('DEPOLARIZE2', targets, [noise_params[CX]])
                    )
        
        return circuit_new
    
    @staticmethod
    def get_transformed_coordinates(circuit: stim.Circuit) -> Dict[int, tuple]:
        """
        获取转换后的探测器坐标
        
        Args:
            circuit: Stim电路
            
        Returns:
            探测器ID到坐标的映射
        """
        d = len(circuit[-1].targets_copy())
        coords = circuit.get_detector_coordinates()
        
        transformed = {
            k: ((2 * d - int(y)) // 2, int(x) // 2, int(z)) 
            for k, (x, y, z) in coords.items()
        }
        
        return transformed

