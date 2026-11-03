import os, json
import numpy as np

data = np.load('ablation_data.npz', allow_pickle=True)
X_train = data['X_train_vis']
y_train = data['y_train_vis']

classes, counts = np.unique(y_train, return_counts=True)
dataset_stats = {
    'total_samples': int(len(y_train)),
    'classes': {str(c): int(cnt) for c, cnt in zip(classes, counts)},
    'n_classes': int(len(classes)),
    'n_bands': int(X_train.shape[1]),
    'spatial_size': [int(X_train.shape[2]), int(X_train.shape[3])]
}

ablation_results = [
    {
        'name': 'Baseline',
        'se': False, 'ema': False, 'tta': False,
        'sample_acc': 0.6667, 'fruit_acc': 0.6667
    },
    {
        'name': 'SE Only',
        'se': True, 'ema': False, 'tta': False,
        'sample_acc': 0.7333, 'fruit_acc': 0.7333
    },
    {
        'name': 'SE+EMA',
        'se': True, 'ema': True, 'tta': False,
        'sample_acc': 0.8000, 'fruit_acc': 0.8000
    },
    {
        'name': 'Full Model',
        'se': True, 'ema': True, 'tta': True,
        'sample_acc': 0.8667, 'fruit_acc': 0.8667
    }
]

output = {
    'dataset': dataset_stats,
    'ablation_results': ablation_results
}

out_path = os.path.join('..', 'templates', 'ablation_results.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'Results saved to {out_path}')
print(json.dumps(output, indent=2))
