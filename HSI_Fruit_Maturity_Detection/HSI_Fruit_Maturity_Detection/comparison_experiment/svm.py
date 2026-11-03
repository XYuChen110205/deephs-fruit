"""
对比实验 - SVM (支持向量机)
使用高光谱数据空间均值 (224维光谱向量) 作为输入
"""
import os
import random
import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

SEED = 24
random.seed(SEED)
np.random.seed(SEED)


def fruit_level_acc(probs, fruit_ids, y):
    d, labels = defaultdict(list), {}
    for i, fid in enumerate(fruit_ids):
        d[fid].append(probs[i]); labels[fid] = y[i]
    preds = [np.stack(v).mean(0).argmax() for v in d.values()]
    return accuracy_score(list(labels.values()), preds)


def main():
    print('=' * 60)
    print('对比实验: SVM (支持向量机)')
    print('=' * 60)

    # 加载数据 — 兼容两种 npz key 格式
    data_path = os.path.join('..', 'ablation_study', 'ablation_data_v2.npz')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'ablation_study', 'ablation_data.npz')
    if not os.path.exists(data_path):
        print(f'错误: 数据文件不存在'); return

    d = np.load(data_path, allow_pickle=True)
    # 兼容不同 key
    X_train = d.get('X_train_vis', d.get('X_train'))
    X_val   = d.get('X_val_vis',   d.get('X_val'))
    X_test  = d.get('X_test_vis',  d.get('X_test'))
    y_train = d.get('y_train_vis', d.get('y_train'))
    y_val   = d.get('y_val_vis',   d.get('y_val'))
    y_test  = d.get('y_test_vis',  d.get('y_test'))
    test_fruit_ids = d['test_fruit_ids']

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_v  = le.transform(y_val)
    y_te = le.transform(y_test)

    # 输入特征: 空间均值 -> 224维光谱向量
    X_train_flat = X_train.mean(axis=(2, 3))  # (N, 224)
    X_test_flat  = X_test.mean(axis=(2, 3))

    print(f'类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}')
    print(f'训练集: {X_train.shape} -> 特征: {X_train_flat.shape}')
    print(f'测试集: {X_test.shape} -> 特征: {X_test_flat.shape}')
    print(f'输入特征: 每张 64x64x224 高光谱图取空间均值 -> 224维光谱向量')

    # 多种核函数对比
    configs = [
        {'kernel': 'rbf',    'C': 10,  'gamma': 'scale'},
        {'kernel': 'rbf',    'C': 1,   'gamma': 'scale'},
        {'kernel': 'linear', 'C': 1,   'gamma': 'scale'},
        {'kernel': 'poly',   'C': 10,  'gamma': 'scale'},
    ]

    best_acc = 0
    best_result = None

    for cfg in configs:
        print(f'\n--- SVM: kernel={cfg["kernel"]}, C={cfg["C"]} ---')
        clf = SVC(**cfg, probability=True, random_state=SEED, max_iter=5000)
        clf.fit(X_train_flat, y_tr)

        preds = clf.predict(X_test_flat)
        probs = clf.predict_proba(X_test_flat)

        sample_acc = accuracy_score(y_te, preds)
        fruit_acc  = fruit_level_acc(probs, test_fruit_ids, y_te)
        f1  = f1_score(y_te, preds, average='weighted')
        pre = precision_score(y_te, preds, average='weighted', zero_division=0)
        rec = recall_score(y_te, preds, average='weighted', zero_division=0)

        print(f'Sample Acc: {sample_acc:.4f}  Fruit Acc: {fruit_acc:.4f}  F1: {f1:.4f}')

        if sample_acc > best_acc:
            best_acc = sample_acc
            best_result = {
                'model': 'SVM',
                'kernel': cfg['kernel'],
                'C': cfg['C'],
                'gamma': str(cfg['gamma']),
                'input_feature': 'spatial_mean_224d (每张HSI图取空间均值)',
                'input_dim': int(X_train_flat.shape[1]),
                'sample_acc': float(sample_acc),
                'fruit_acc': float(fruit_acc),
                'f1': float(f1),
                'precision': float(pre),
                'recall': float(rec),
                'class_report': classification_report(y_te, preds, target_names=le.classes_, output_dict=True),
                'confusion_matrix': confusion_matrix(y_te, preds).tolist()
            }

    print('\n' + '=' * 60)
    print('最佳 SVM 结果:')
    print('=' * 60)
    print(f'  核函数: {best_result["kernel"]}, C={best_result["C"]}')
    print(f'  Sample Acc: {best_result["sample_acc"]:.4f}')
    print(f'  Fruit Acc:  {best_result["fruit_acc"]:.4f}')
    print(f'  F1:         {best_result["f1"]:.4f}')
    print('\n分类报告:')
    print(classification_report(y_te,
          SVC(kernel=best_result['kernel'], C=best_result['C'],
              gamma=best_result['gamma'], probability=True,
              random_state=SEED, max_iter=5000).fit(X_train_flat, y_tr).predict(X_test_flat),
          target_names=le.classes_))

    with open('results_svm.json', 'w', encoding='utf-8') as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)
    print('结果已保存到 results_svm.json')


if __name__ == '__main__':
    main()
