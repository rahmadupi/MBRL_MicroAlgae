"""Atomic epoch-boundary recovery, separate from inference and warm-start artifacts."""

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()


def recovery_identity(wm, algorithm, index, validation, preprocessing, config, device):
    stores = {str(k): {name: array_digest(store[name])
                       for name in ['values', 'targets', 'timestamps', 'observed']}
              for k, store in index.stores.items()}
    validation_digest = {}
    for name, prepared in [('checkpoint', validation), ('treatment', (validation or {}).get('treatment_diagnostic'))]:
        if prepared is not None and len(prepared.get('y', [])):
            validation_digest[name] = {key: array_digest(prepared['raw'][key])
                                       for key in ['sequence', 'extra', 'truth', 'latest']}
            validation_digest[name]['anchors'] = digest(prepared['raw']['anchors'])
    return {'algorithm': algorithm, 'architecture': wm.make_architecture(algorithm, config),
            'settings': config['neural'][algorithm], 'preprocessing': preprocessing,
            'training_entries': array_digest(index.entries), 'training_stores': stores,
            'validation': validation_digest, 'seed': config['seed'],
            'source_manifest': config.get('source_manifest'),
            'runtime': {'device': str(device), 'torch': str(wm.torch.__version__),
                        'cuda': wm.torch.version.cuda, 'cpu_threads': config['cpu_threads'],
                        'loader_workers': config['loader_workers'],
                        'validation_batch_size': config['validation_batch_size']},
            'backend_sha256': wm.file_hash(wm.__file__),
            'protocol_sha256': wm.file_hash(wm.wp.__file__),
            'recovery_sha256': wm.file_hash(__file__)}


def capture_rng(torch, generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(torch, generator, state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_copy(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(torch, v) for v in value)
    return value


def atomic_save(torch, payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as handle:
            torch.save(cpu_copy(torch, payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(torch, path, identity):
    # Recovery files contain Python/NumPy RNG states; only load a trusted local file.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != 'world_model_recovery_v1':
        raise ValueError('Not a full-state recovery checkpoint; inference/warm-start files are different.')
    if digest(payload['identity']) != digest(identity):
        raise ValueError('Recovery mismatch: data, windows, preprocessing, settings, code or runtime changed. '
                         'Restore the original configuration or start a new experiment.')
    return payload
