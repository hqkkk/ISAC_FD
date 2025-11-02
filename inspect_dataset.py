import torch
import os
import traceback

path = os.path.join(os.path.dirname(__file__), 'dataset', 'dataset.pt')
print('Inspecting', path)
if not os.path.exists(path):
    print('File not found:', path)
    raise SystemExit(1)

# Try to load; prefer weights_only=False so we can inspect objects. Only do this locally when you trust the file.
try:
    data = torch.load(path, map_location='cpu', weights_only=False)
except Exception as e:
    print('Initial load failed with exception:')
    traceback.print_exc()
    # Try again with a safe globals allowlist for common classes
    try:
        print('\nRetrying with TensorDataset allowed in safe globals...')
        torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
        data = torch.load(path, map_location='cpu')
    except Exception:
        print('Retry also failed:')
        traceback.print_exc()
        raise

print('\nLoaded object type:', type(data))

if isinstance(data, dict):
    print('Dictionary keys:', list(data.keys()))
    for k,v in data.items():
        try:
            if hasattr(v, 'shape'):
                print(f"  {k}: type={type(v)}, shape={getattr(v,'shape',None)}, dtype={getattr(v,'dtype',None)}")
            else:
                print(f"  {k}: type={type(v)}")
        except Exception:
            print(f"  {k}: (could not introspect)")
elif hasattr(data, 'tensors'):
    print('Object has .tensors attribute (likely TensorDataset). Number of tensors:', len(data.tensors))
    for i,t in enumerate(data.tensors):
        try:
            print(f'  tensor[{i}]: shape={t.shape}, dtype={t.dtype}')
        except Exception:
            print(f'  tensor[{i}]: (could not introspect)')
elif isinstance(data, (list, tuple)):
    print('List/Tuple of length', len(data))
    for i,elem in enumerate(data):
        try:
            if hasattr(elem,'shape'):
                print(f'  elem[{i}]: type={type(elem)}, shape={elem.shape}, dtype={getattr(elem,"dtype",None)}')
            else:
                print(f'  elem[{i}]: type={type(elem)}')
        except Exception:
            print(f'  elem[{i}]: (could not introspect)')
else:
    # fallback: try to print a short repr
    try:
        r = repr(data)
        print('Repr (first 1000 chars):\n', r[:1000])
    except Exception:
        print('Cannot repr the loaded object')

print('\nDone')
