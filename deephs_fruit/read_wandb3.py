import re, os

wandb_dir = os.path.join(os.getcwd(),'wandb')
for run in sorted(os.listdir(wandb_dir)):
    path = os.path.join(wandb_dir,run)
    if not os.path.isdir(path): continue
    parts = run.split('-')
    suffix = parts[-1]
    wf = os.path.join(path,'run-'+suffix+'.wandb')
    if not os.path.exists(wf): continue
    
    with open(wf,'rb') as f:
        data = f.read()
    
    # Find num_epochs
    epochs = '?'
    mr = re.search(rb'num_epochs\*(\d+)', data)
    if mr: epochs = mr.group(1).decode()
    
    # Find confusion matrix epoch
    cm_epoch = '?'
    mr = re.search(rb'confusion_matrix_(\d+)', data)
    if mr: cm_epoch = mr.group(1).decode()
    
    # Find test accuracy from summary at end
    # look for "test/accuracy" then extract float near it
    acc = '?'
    loss_val = '?'
    
    # The summary metrics are stored as protobuf near the end
    # Look for the specific float64 values stored for test/accuracy and test/loss
    idx = data.rfind(b'\n'+b'test/accuracy')
    if idx < 0: idx = data.rfind(b'test/accuracy')
    if idx > 0:
        chunk = data[idx:idx+300]
        for m in re.finditer(rb'(\d+\.\d+e?[+-]?\d*)', chunk):
            val = float(m.group())
            if 0.0 < val <= 1.0:
                acc = val
    
    idx = data.rfind(b'\n'+b'test/loss')
    if idx < 0: idx = data.rfind(b'test/loss')
    if idx > 0:
        chunk = data[idx:idx+300]
        for m in re.finditer(rb'(\d+\.\d+e?[+-]?\d*)', chunk):
            val = float(m.group())
            if 1.0 < val <= 10.0:
                loss_val = val
    
    # Also find val/accuracy for comparison
    vacc = '?'
    idx = data.rfind(b'val/accuracy')
    if idx > 0:
        chunk = data[idx:idx+200]
        for m in re.finditer(rb'(\d+\.\d+)', chunk):
            val = float(m.group())
            if 0.0 < val <= 1.0:
                vacc = max(vacc, val) if isinstance(vacc, float) else val
    
    print(f'{run}')
    print(f'  epochs={epochs} cm_epoch={cm_epoch} test/acc={acc} test/loss={loss_val} val/acc={vacc}')
    print()
