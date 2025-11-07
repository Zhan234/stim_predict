"""结果可视化工具"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def visualize_distribution_distance(results: Dict[str, Any]):
    """可视化概率分布距离结果"""
    print_separator()
    print("概率分布距离评测结果")
    print_separator()
    
    # 提取方法列表
    methods = [k for k in results.keys() if k != 'rankings']
    
    # 创建表格
    print(f"\n{'方法':<20} {'MAE':<12} {'RMSE':<12} {'KL散度':<12} {'相关系数':<12}")
    print("-" * 80)
    
    for method in methods:
        if 'metrics' in results[method]:
            metrics = results[method]['metrics']
            mae = metrics.get('mean_absolute_error', 0)
            rmse = metrics.get('root_mean_squared_error', 0)
            kl = metrics.get('kl_divergence', 0)
            corr = metrics.get('correlation', 0)
            
            print(f"{method:<20} {mae:<12.6f} {rmse:<12.6f} {kl:<12.6f} {corr:<12.6f}")
    
    # 显示排名
    if 'rankings' in results:
        print("\n排名 (越小越好):")
        print("-" * 80)
        for metric, ranking in results['rankings'].items():
            print(f"  {metric}: {' > '.join(ranking)}")


def visualize_decoder_ler(results: Dict[str, Any]):
    """可视化解码器LER结果"""
    print_separator()
    print("解码器逻辑错误率评测结果")
    print_separator()
    
    # 提取方法列表
    methods = [k for k in results.keys() if k != 'rankings']
    
    # 按解码器分组显示
    if methods:
        first_method = methods[0]
        if 'decoders' in results[first_method]:
            decoders = list(results[first_method]['decoders'].keys())
            
            for decoder in decoders:
                print(f"\n解码器: {decoder}")
                print("-" * 80)
                print(f"{'方法':<20} {'预测DEM LER':<15} {'真实DEM LER':<15} {'比率':<10}")
                print("-" * 80)
                
                for method in methods:
                    if decoder in results[method].get('decoders', {}):
                        decoder_result = results[method]['decoders'][decoder]
                        if 'predicted_dem' in decoder_result:
                            pred_ler = decoder_result['predicted_dem']['ler']
                            true_ler = decoder_result['ground_truth_dem']['ler']
                            ratio = decoder_result['ler_ratio']
                            
                            print(f"{method:<20} {pred_ler:<15.6f} {true_ler:<15.6f} {ratio:<10.3f}")
    
    # 显示排名
    if 'rankings' in results:
        print("\n排名 (按LER，越小越好):")
        print("-" * 80)
        for decoder_name, ranking in results['rankings'].items():
            print(f"\n  {decoder_name}:")
            for i, item in enumerate(ranking, 1):
                print(f"    {i}. {item['method']}: LER={item['ler']:.6f}")


def visualize_experiment(experiment_name: str, data_dir: str = "./stim_predict/data"):
    """可视化整个实验的结果"""
    exp_path = Path(data_dir) / experiment_name
    
    if not exp_path.exists():
        print(f"错误: 实验 '{experiment_name}' 不存在")
        return
    
    print_separator('=')
    print(f"实验: {experiment_name}")
    print_separator('=')
    
    # 读取元数据
    metadata_path = exp_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\n实验配置:")
        print("-" * 80)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # 读取评测结果
    eval_dir = exp_path / "evaluations"
    if eval_dir.exists():
        # 概率分布距离
        dist_path = eval_dir / "distribution_distance.json"
        if dist_path.exists():
            with open(dist_path, 'r') as f:
                dist_results = json.load(f)
            print()
            visualize_distribution_distance(dist_results)
        
        # 解码器LER
        ler_path = eval_dir / "decoder_ler.json"
        if ler_path.exists():
            with open(ler_path, 'r') as f:
                ler_results = json.load(f)
            print()
            visualize_decoder_ler(ler_results)
    else:
        print("\n警告: 未找到评测结果")
    
    print()
    print_separator('=')


def list_experiments(data_dir: str = "./stim_predict/data"):
    """列出所有实验"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print("数据目录不存在")
        return
    
    experiments = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    if not experiments:
        print("没有找到任何实验")
        return
    
    print_separator()
    print("可用的实验:")
    print_separator()
    
    for exp_name in experiments:
        exp_path = data_path / exp_name
        
        # 读取元数据
        metadata_path = exp_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            code_type = metadata.get('code_type', 'N/A')
            distance = metadata.get('distance', 'N/A')
            n_shots = metadata.get('n_shots', 'N/A')
            
            print(f"  - {exp_name}")
            print(f"      类型: {code_type}, 码距: {distance}, 采样: {n_shots}")
        else:
            print(f"  - {exp_name}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='可视化评测结果')
    
    parser.add_argument('--experiment', type=str,
                       help='要可视化的实验名称')
    parser.add_argument('--list', action='store_true',
                       help='列出所有实验')
    parser.add_argument('--data-dir', type=str, default='./stim_predict/data',
                       help='数据目录路径')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments(args.data_dir)
    elif args.experiment:
        visualize_experiment(args.experiment, args.data_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

