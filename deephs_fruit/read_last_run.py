import re, os

wandb_dir = os.path.join(os.getcwd(),'wandb')

# Latest run - r3dqe5ev
wf = os.path.join(wandb_dir, 'offline-run-20260424_225816-r3dqe5ev', 'run-r3dqe5ev.wandb')
with open(wf,'rb') as f:
    data = f.read()

# Find ALL numeric values near test/accuracy and test/loss at the END
tail = data[-10000:]

# Search backwards for test metrics
for pos in range(len(tail)-1, 0, -1):
    if tail[pos:pos+13] == b'test/accuracy' or tail[pos:pos+5] == b'test/':
        chunk = tail[max(0,pos-50):pos+300]
        readable = []
        for m in re.finditer(rb'[\x20-\x7e]+', chunk):
            t = m.group().decode('ascii',errors='replace')
            readable.append(t)
        if 'accuracy' in ''.join(readable) or 'test' in ''.join(readable):
            print(f'At tail offset {pos}:')
            print(' '.join(readable[-20:]))
            print()

# Also search for val/accuracy
for pos in range(len(tail)-1, 0, -1):
    if tail[pos:pos+12] == b'val/accuracy':
        chunk = tail[max(0,pos-50):pos+300]
        readable = []
        for m in re.finditer(rb'[\x20-\x7e]+', chunk):
            t = m.group().decode('ascii',errors='replace')
            readable.append(t)
        print(f'val/accuracy at offset {pos}:')
        print(' '.join(readable[-15:]))
        print()

# Check confusion matrix
for m in re.finditer(rb'confusion_matrix_(\d+)', data):
    print(f'confusion matrix at step: {m.group(1).decode()}')

# Print last 2000 bytes as ASCII for manual inspection
print("=== Last 1000 bytes (printable) ===")
last_bytes = data[-2000:]
for m in re.finditer(rb'[\x20-\x7e]{3,}', last_bytes):
    print(m.group().decode('ascii',errors='replace'))
