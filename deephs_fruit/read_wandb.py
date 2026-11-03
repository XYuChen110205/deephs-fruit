import os, re

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

    texts = []
    for m in re.finditer(rb'[\x20-\x7e]{4,}', data):
        t = m.group().decode('ascii',errors='replace')
        texts.append(t)

    info = run.split('_')
    date = info[1] if len(info)>1 else '?'
    timestr = info[2][:6] if len(info)>2 else '?'

    print('=== Run:', date, timestr, '('+suffix+') ===')

    test_texts = [t for t in texts if 'test' in t.lower()]
    for t in test_texts[:10]:
        print(' ', t)

    idx = data.rfind(b'test')
    if idx > 0:
        start = max(0,idx-50)
        end = min(len(data),idx+300)
        snippet = data[start:end]
        readable = []
        for m in re.finditer(rb'[\x20-\x7e]+', snippet):
            readable.append(m.group().decode('ascii',errors='replace'))
        print('  Around test:', readable)
    print()
