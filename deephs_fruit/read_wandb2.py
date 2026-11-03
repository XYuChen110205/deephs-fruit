import os, re, struct

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

    # Find test/accuracy and test/loss in binary
    for key in [b'test/accuracy', b'test/loss', b'val/accuracy', b'val/loss', b'train/accuracy']:
        idx = data.rfind(key)
        if idx > 0:
            # Look at bytes after the key
            after = data[idx+len(key):idx+50]
            printable = ''.join(chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b in after[:30])
            print(f'{run}: {key.decode()} -> {printable}')
