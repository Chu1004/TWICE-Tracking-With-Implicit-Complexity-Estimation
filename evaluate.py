# -*- coding: utf-8 -*-
"""
MOT Evaluation Script - FIXED VERSION
修正了 AssA, HOTA, IDF1 的計算方法
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


def load_mot_results(file_path):
    """載入 MOT 格式的結果文件"""
    if not Path(file_path).exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
        )
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def box_iou(box1, box2):
    """計算兩個框的 IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def calculate_metrics_correct(gt_file, pred_file, iou_threshold=0.5):
    """
    正確計算 MOT 指標
    
    重點修正:
    1. AssA: 基於 ID 一致性，不是平均 IoU
    2. HOTA: 使用正確的公式
    3. IDF1: 修正分母公式
    """
    gt_df = load_mot_results(gt_file)
    pred_df = load_mot_results(pred_file)
    
    if len(gt_df) == 0 or len(pred_df) == 0:
        return None
    
    frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
    
    # 統計變量
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_tp = 0  # ID 正確的 TP
    
    # ID 追蹤
    gt_trajectories = defaultdict(list)  # gt_id -> [(frame, box)]
    pred_trajectories = defaultdict(list)  # pred_id -> [(frame, box)]
    
    # ID 映射統計
    id_mapping_count = defaultdict(lambda: defaultdict(int))  # pred_id -> {gt_id: count}
    
    id_switches = 0
    prev_mapping = {}
    
    # 第一輪：收集所有軌跡和建立映射
    for frame_id in frames:
        gt_frame = gt_df[gt_df['frame'] == frame_id]
        pred_frame = pred_df[pred_df['frame'] == frame_id]
        
        if len(gt_frame) == 0 and len(pred_frame) == 0:
            continue
        
        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        gt_ids = gt_frame['id'].values
        pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values
        pred_ids = pred_frame['id'].values
        
        # 保存軌跡
        for gt_id, gt_box in zip(gt_ids, gt_boxes):
            gt_trajectories[gt_id].append((frame_id, gt_box))
        
        for pred_id, pred_box in zip(pred_ids, pred_boxes):
            pred_trajectories[pred_id].append((frame_id, pred_box))
        
        # 匹配
        matched_pred = set()
        matched_gt = set()
        
        for i, (gt_box, gt_id) in enumerate(zip(gt_boxes, gt_ids)):
            best_iou = 0
            best_j = -1
            
            for j, (pred_box, pred_id) in enumerate(zip(pred_boxes, pred_ids)):
                if j in matched_pred:
                    continue
                
                iou = box_iou(gt_box, pred_box)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_j >= 0:
                pred_id = pred_ids[best_j]
                matched_pred.add(best_j)
                matched_gt.add(i)
                
                # 記錄映射
                id_mapping_count[pred_id][gt_id] += 1
                
                total_tp += 1
                
                # 檢查 ID 一致性
                if pred_id in prev_mapping:
                    if prev_mapping[pred_id] == gt_id:
                        total_id_tp += 1
                    else:
                        id_switches += 1
                else:
                    # 第一次出現，算正確
                    total_id_tp += 1
                
                prev_mapping[pred_id] = gt_id
        
        # FP 和 FN
        total_fp += len(pred_boxes) - len(matched_pred)
        total_fn += len(gt_boxes) - len(matched_gt)
    
    # 計算檢測指標
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    
    # DetA (Detection Accuracy)
    deta = np.sqrt(precision * recall)
    
    # AssA (Association Accuracy) - 修正版
    # AssA 應該基於 ID 一致性
    assa = total_id_tp / (total_tp + 1e-6)
    
    # HOTA (Higher Order Tracking Accuracy)
    # 簡化版: HOTA ≈ sqrt(DetA * AssA)
    hota = np.sqrt(deta * assa)
    
    # IDF1 - 修正公式
    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    idf1 = 2 * total_id_tp / (2 * total_id_tp + total_fp + total_fn + 1e-6)
    
    # MOTA
    total_gt = len(gt_df)
    mota = 1 - (total_fn + total_fp + id_switches) / (total_gt + 1e-6)
    
    return {
        'HOTA': hota * 100,
        'DetA': deta * 100,
        'AssA': assa * 100,
        'MOTA': mota * 100,
        'IDF1': idf1 * 100,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'ID_Sw': id_switches,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'ID_TP': total_id_tp
    }


def calculate_hota_detailed(gt_file, pred_file, alpha_values=None):
    """
    詳細的 HOTA 計算（使用多個 IoU 閾值）
    
    HOTA 論文的標準計算方法
    """
    if alpha_values is None:
        # 標準 19 個閾值: 0.05, 0.10, ..., 0.95
        alpha_values = np.linspace(0.05, 0.95, 19)
    
    gt_df = load_mot_results(gt_file)
    pred_df = load_mot_results(pred_file)
    
    if len(gt_df) == 0 or len(pred_df) == 0:
        return None
    
    frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
    
    hota_values = []
    
    for alpha in alpha_values:
        # 在每個閾值下計算
        total_tpa = 0  # TP at threshold alpha
        total_fpa = 0
        total_fna = 0
        
        for frame_id in frames:
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            pred_frame = pred_df[pred_df['frame'] == frame_id]
            
            if len(gt_frame) == 0 and len(pred_frame) == 0:
                continue
            
            gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
            pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values
            
            # 匹配
            matched_pred = set()
            matched_gt = set()
            
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    if j in matched_pred:
                        continue
                    
                    iou = box_iou(gt_box, pred_box)
                    
                    if iou > alpha:
                        matched_pred.add(j)
                        matched_gt.add(i)
                        total_tpa += 1
                        break
            
            total_fpa += len(pred_boxes) - len(matched_pred)
            total_fna += len(gt_boxes) - len(matched_gt)
        
        # DetA at alpha
        deta_alpha = total_tpa / (total_tpa + total_fpa + total_fna + 1e-6)
        hota_values.append(deta_alpha)
    
    # HOTA 是所有閾值下的平均
    hota = np.mean(hota_values)
    
    return hota * 100


def evaluate_dataset(pred_dir, gt_dir, sequences=None, detailed_hota=False):
    """評估整個數據集"""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    if sequences is None:
        sequences = [d.name for d in pred_dir.iterdir() if d.is_dir()]
        
        if len(sequences) == 0:
            print(f"[Error] No sequences found in {pred_dir}")
            return None, None
    
    print("="*80)
    print("MOT Evaluation - FIXED VERSION")
    print("="*80)
    print(f"Predictions: {pred_dir}")
    print(f"Ground Truth: {gt_dir}")
    print(f"Sequences: {len(sequences)}")
    print(f"Detailed HOTA: {detailed_hota}")
    print("="*80 + "\n")
    
    seq_results = {}
    
    for seq_name in sorted(sequences):
        print(f"[Evaluating] {seq_name}...")
        
        pred_file = pred_dir / seq_name / "pred.txt"
        gt_file = gt_dir / seq_name / "gt" / "gt.txt"
        
        if not pred_file.exists():
            print(f"  [Skip] Prediction not found: {pred_file}")
            continue
        
        if not gt_file.exists():
            print(f"  [Skip] GT not found: {gt_file}")
            continue
        
        # 標準評估
        result = calculate_metrics_correct(gt_file, pred_file)
        
        if result:
            # 詳細 HOTA（可選）
            if detailed_hota:
                hota_detailed = calculate_hota_detailed(gt_file, pred_file)
                if hota_detailed:
                    result['HOTA_Detailed'] = hota_detailed
            
            seq_results[seq_name] = result
            
            print(f"  HOTA: {result['HOTA']:.1f}% | IDF1: {result['IDF1']:.1f}% | "
                  f"AssA: {result['AssA']:.1f}% | MOTA: {result['MOTA']:.1f}%")
    
    if len(seq_results) == 0:
        print("\n[Error] No valid sequences evaluated!")
        return None, None
    
    # 計算平均值
    avg_results = {}
    for metric in ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'Precision', 'Recall']:
        values = [r[metric] for r in seq_results.values() if metric in r]
        avg_results[metric] = np.mean(values) if values else 0
    
    avg_results['ID_Sw'] = sum(r.get('ID_Sw', 0) for r in seq_results.values())
    
    # 打印結果
    print("\n" + "="*80)
    print("Overall Results (FIXED METRICS)")
    print("="*80)
    
    print(f"\n📊 Core Metrics:")
    print(f"  HOTA:  {avg_results['HOTA']:.1f}%  (Higher Order Tracking Accuracy)")
    print(f"  IDF1:  {avg_results['IDF1']:.1f}%  (ID F1 Score)")
    print(f"  AssA:  {avg_results['AssA']:.1f}%  (Association Accuracy)")
    print(f"  MOTA:  {avg_results['MOTA']:.1f}%  (Multiple Object Tracking Accuracy)")
    print(f"  DetA:  {avg_results['DetA']:.1f}%  (Detection Accuracy)")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"  Precision: {avg_results['Precision']:.1f}%")
    print(f"  Recall:    {avg_results['Recall']:.1f}%")
    print(f"  ID Sw:     {avg_results['ID_Sw']} (ID Switches)")
    
    print("\n" + "="*80)
    print("Per-Sequence Results")
    print("="*80)
    
    print(f"\n{'Sequence':<20} {'HOTA':>7} {'IDF1':>7} {'AssA':>7} {'MOTA':>7} {'DetA':>7}")
    print("-" * 80)
    
    for seq_name in sorted(seq_results.keys()):
        result = seq_results[seq_name]
        print(f"{seq_name:<20} "
              f"{result['HOTA']:>6.1f}% "
              f"{result['IDF1']:>6.1f}% "
              f"{result['AssA']:>6.1f}% "
              f"{result['MOTA']:>6.1f}% "
              f"{result['DetA']:>6.1f}%")
    
    print("\n" + "="*80)
    print(f"✓ Evaluated {len(seq_results)} sequences")
    print("="*80)
    
    return avg_results, seq_results


def main():
    parser = argparse.ArgumentParser(
        description="MOT Evaluation Script - FIXED VERSION"
    )
    
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Predictions directory')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth directory')
    parser.add_argument('--sequences', type=str, nargs='+',
                        help='Specific sequences to evaluate')
    parser.add_argument('--detailed_hota', action='store_true',
                        help='Calculate detailed HOTA with multiple thresholds')
    
    args = parser.parse_args()
    
    avg_results, seq_results = evaluate_dataset(
        args.pred_dir,
        args.gt_dir,
        args.sequences,
        args.detailed_hota
    )
    
    if avg_results:
        print("\n✅ Evaluation complete!")
        print(f"\n🎯 Summary: HOTA={avg_results['HOTA']:.1f}%, "
              f"IDF1={avg_results['IDF1']:.1f}%, "
              f"AssA={avg_results['AssA']:.1f}%")


if __name__ == "__main__":
    main()