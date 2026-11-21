"""Within-recording development pilot for calibration f558fd22."""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

from ketik_audio.audio import extract
from ketik_audio.common import FeatureConfig, file_hash, fresh_directory, read_json, write_json
from ketik_audio.skaid_pilot_features import mfcc_features
from ketik_audio.web_alignment import consensus_timing
from ketik_audio.web_collection import coverage, read_events, replay_timing
from ketik_audio.web_quota import CLASSES

RUN = Path(__file__).resolve().parent
SOURCE = RUN.parents[1] / 'data/collected/lenovo-loq-calibration-f558fd22'
ROLES = ('train', 'validation', 'test')
SEEDS = (42, 43, 44)
LABEL_IDS = np.arange(len(CLASSES))
BLOCK_SECONDS = 8
EDGE_SECONDS = 0.15
FEATURES = FeatureConfig(sample_rate=16000, n_fft=256, hop_length=80, n_mels=64,
                         n_mfcc=13, segment_ms=80, pre_ms=15)


def relative_energy(audio, rate):
    size = round(.005 * rate)
    bins = audio[:len(audio) // size * size].reshape(-1, size)
    rms = bins.std(axis=1, dtype=np.float64)
    centers = (np.arange(len(rms)) + .5) * size
    quiet = rms[round(.5 * rate / size):round(2.5 * rate / size)]
    floor = float(np.median(quiet))
    p95 = np.sort(quiet)[int(.95 * (len(quiet) - 1))]
    return centers, rms, floor, float(max(1e-6, floor * 10 ** .3, 1.2 * p95))


def pilot_alignment(events, clock, centers, rms, floor, threshold, rate):
    markers = []
    for event in events:
        if not event.get('calibration'):
            continue
        at = clock(event['time_ms'])
        if at is None:
            raise ValueError('Penanda pemeriksaan di luar rentang jam audio.')
        lo, hi = np.searchsorted(centers, [at - .065 * rate, at + .065 * rate])
        index = lo + int(rms[lo:hi].argmax())
        distant = np.abs(centers[lo:hi] - centers[index]) > .025 * rate
        reason = None
        if rms[index] <= threshold:
            reason = 'weak_signal'
        elif np.any(rms[lo:hi][distant] >= .9 * rms[index]):
            reason = 'ambiguous_peak'
        markers.append({'event_seq': event['seq'], 'phase': event['phase'], 'reason': reason,
                        'clock_frame': at, 'peak_frame': float(centers[index]),
                        'snr_db': float(20 * np.log10(rms[index] / floor))})
    fit = consensus_timing({'calibration': markers, 'issues': []}, rate)
    fit['method'] = 'pilot-ac-rms-5ms-local-peak-affine'
    fit['peak_search_ms'] = [-65, 65]
    fit['quiet_ac_rms'] = floor
    fit['relative_threshold_rms'] = threshold
    if fit['status'] != 'checked':
        raise ValueError(f"Uji timing pilot gagal: {fit['issues']}")
    return fit


def split_blocks(blocks, labels):
    block_ids, inverse = np.unique(blocks, return_inverse=True)
    counts = np.zeros((len(block_ids), len(CLASSES)), dtype=int)
    np.add.at(counts, (inverse, labels), 1)
    target = np.asarray([.6, .2, .2])[:, None]
    minimum = np.asarray([5, 2, 2])[:, None]
    n_held = round(len(block_ids) / 5)
    rng = np.random.default_rng(42)
    best = None
    best_loss = np.inf
    for _ in range(50_000):
        permutation = rng.permutation(len(block_ids))
        groups = np.split(permutation, [len(block_ids) - 2 * n_held, len(block_ids) - n_held])
        totals = np.stack([counts[group].sum(0) for group in groups])
        if (totals < minimum).any():
            continue
        loss = float(np.abs(totals / counts.sum(0) - target).mean())
        if loss < best_loss:
            best, best_loss = groups, loss
    if best is None:
        raise ValueError('Pembagian blok tidak mencakup seluruh kelas di tiga bagian.')
    role_by_block = {int(block_ids[g]): role for role, group in zip(ROLES, best) for g in group}
    fold = np.array([role_by_block[int(block)] for block in blocks])
    return fold, role_by_block


def prepare(source):
    output = RUN / 'prepared'
    fresh_directory(output)
    meta = read_json(source / 'session.json')
    review = read_json(source / 'review.json')
    original = read_json(source / 'alignment-v2.json')
    audio, rate = sf.read(source / 'audio.wav', dtype='float32')
    events = read_events((source / 'events.jsonl').read_bytes(), meta['event_count'])
    if rate != 48000 or audio.ndim != 1 or review['capture_status'] != 'complete':
        raise ValueError('Pilot ini memakai WAV mono 48 kHz dari rekaman lengkap.')
    _, clock = replay_timing(audio, rate, meta['first_frame'], events,
                             read_json(source / 'timing.json'), meta['capture_issues'])
    centers, rms, floor, threshold = relative_energy(audio, rate)
    fit = pilot_alignment(events, clock, centers, rms, floor, threshold, rate)
    write_json(output / 'pilot-timing.json', fit)
    selected = meta['collection']['selected_sequences']
    isolated = coverage(audio, rate, events, fit, clock, CLASSES, selected)
    rejects = Counter(isolated['rejected'])
    rows, crops = [], []
    for item in isolated['candidate_events']:
        at = item['frame_estimate']
        lo, hi = np.searchsorted(centers, [at - .015 * rate, at + .035 * rate])
        if rms[lo:hi].max() <= threshold:
            rejects['weak_local_transient'] += 1
            continue
        seconds = at / rate
        block = int(seconds // BLOCK_SECONDS)
        if not EDGE_SECONDS <= seconds % BLOCK_SECONDS <= BLOCK_SECONDS - EDGE_SECONDS:
            rejects['block_guard'] += 1
            continue
        start, end = round(at - .015 * rate), round(at + .065 * rate)
        seq = item['event_seq']
        rows.append({'event_seq': seq, 'key': item['letter'], 'block': block,
                     'center_s': seconds, 'start_frame': start, 'end_frame': end,
                     'prompt': events[seq]['collection_mode']})
        crops.append(audio[start:end])

    labels = np.array([CLASSES.index(row['key']) for row in rows])
    blocks = np.array([row['block'] for row in rows])
    fold, roles = split_blocks(blocks, labels)
    for row, role in zip(rows, fold):
        row['split'] = role
    ordered = sorted(rows, key=lambda row: row['start_frame'])
    if any(left['end_frame'] > right['start_frame'] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Terdapat potongan audio bertumpang tindih.')
    if len(rows) + sum(rejects.values()) != len(selected):
        raise ValueError('Total potongan tidak cocok dengan total kuota.')
    with (output / 'split.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {role: {key: int(np.sum((labels == idx) & (fold == role)))
                     for idx, key in enumerate(CLASSES)} for role in ROLES}
    summary = {
        'source': str(source.resolve()), 'source_sha256': review['source_sha256'],
        'script_sha256': file_hash(Path(__file__)), 'session': meta['session_id'],
        'duration_seconds': len(audio) / rate, 'capture_status': review['capture_status'],
        'standard_timing': original['status'], 'standard_timing_issues': original['issues'],
        'pilot_max_validation_residual_ms': fit['max_abs_validation_residual_ms'],
        'pilot_excluded_fit_sequences': fit['excluded_training_peaks'],
        'quiet_device_dbfs': float(20 * np.log10(meta['preflight']['quiet_raw_rms'])),
        'qa_initial_taps': meta['preflight']['passed'], 'quality': review['quality'],
        'recorded_counts': meta['collection']['raw_counts'], 'selected_count': len(selected),
        'additional_keys': review['additional_keys']['counts'],
        'isolated_count': len(isolated['candidate_events']), 'used_count': len(rows),
        'rejected': dict(rejects), 'split_counts': counts,
        'split_sizes': {role: int(np.sum(fold == role)) for role in ROLES},
        'split_blocks': roles, 'split_block_seconds': BLOCK_SECONDS,
        'split_guard_seconds': EDGE_SECONDS, 'cnn_features': FEATURES.to_dict(),
        'mfcc_features': 'SKAID MFCC78, mono 16000 Hz and 80 ms',
    }
    write_json(output / 'dataset.json', summary)
    print({k: summary[k] for k in ('used_count', 'rejected', 'split_sizes')}, flush=True)
    waves = resample_poly(np.stack(crops), up=1, down=3, axis=1).astype(np.float32)
    mfcc = mfcc_features(waves)
    mel = np.stack([extract(wave, FEATURES)[0] for wave in waves])
    np.savez_compressed(output / 'features.npz', mfcc=mfcc, mel=mel, labels=labels,
                        fold=fold, blocks=blocks, seq=np.array([row['event_seq'] for row in rows]),
                        prompt=np.array([row['prompt'] for row in rows]))
    print(f'Fitur siap: {mfcc.shape}, log-mel {mel.shape}', flush=True)


def measure(labels, scores):
    order = np.argsort(-scores, axis=1, kind='stable')
    predicted = order[:, 0]
    return {'n': len(labels), 'correct': int(np.sum(labels == predicted)),
            'accuracy': float(accuracy_score(labels, predicted)),
            'macro_f1': float(f1_score(labels, predicted, labels=LABEL_IDS, average='macro',
                                      zero_division=0)),
            'top3': float(np.mean(np.any(order[:, :3] == labels[:, None], axis=1)))}


def classical(features, labels, masks, output):
    ridge = [(f'alpha={alpha}', make_pipeline(StandardScaler(), Normalizer(), RidgeClassifier(alpha=alpha)))
             for alpha in (.01, .1, 1., 10.)]
    svm = [(f'C={c}, gamma={gamma}', make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma)))
           for c in (1, 10, 100) for gamma in ('scale', .01)]
    results, test_scores = {}, {}
    for name, candidates in [('mfcc-ridge', ridge), ('mfcc-rbf', svm)]:
        fitted = []
        for recipe, model in candidates:
            model.fit(features[masks['train']], labels[masks['train']])
            val = measure(labels[masks['validation']], model.decision_function(features[masks['validation']]))
            fitted.append((recipe, model, val))
        recipe, model, val = max(fitted, key=lambda item: item[2]['macro_f1'])
        joblib.dump(model, output / f'{name}.joblib')
        scores = model.decision_function(features[masks['test']])
        results[name] = {'selection': recipe, 'validation': val,
                         'validation_grid': [{'recipe': r, **m} for r, _, m in fitted],
                         'test': measure(labels[masks['test']], scores)}
        test_scores[name] = scores
        print(name, recipe, results[name]['test'], flush=True)

    control = []
    train, test = masks['train'], masks['test']
    rng = np.random.default_rng(102)
    for _ in range(20):
        model = make_pipeline(StandardScaler(), Normalizer(), RidgeClassifier(alpha=1))
        model.fit(features[train], rng.permutation(labels[train]))
        control.append(measure(labels[test], model.decision_function(features[test])))
    results['shuffled-ridge'] = {'test': {metric: float(np.mean([m[metric] for m in control]))
                                         for metric in ('accuracy', 'macro_f1', 'top3')},
                                 'permutation_runs': control}
    return results, test_scores


def residual_cnn(mel, labels, masks, output, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from ketik_audio.cnn_repair import repaired_model
    from ketik_audio.models import _cnn_scores, _seed_everything

    train, val, test = (masks[role] for role in ROLES)
    mean = mel[train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(mel[train].std(axis=(0, 2), keepdims=True), 1e-6)
    x = torch.from_numpy(((mel - mean) / std)[:, None].astype(np.float32))
    y = torch.from_numpy(labels).long()
    dataset = TensorDataset(x[train], y[train])
    frequencies = np.bincount(labels[train], minlength=len(CLASSES))
    weights = torch.as_tensor(frequencies.mean() / frequencies, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.03)
    result, probabilities = [], {}
    for seed in SEEDS:
        _seed_everything(seed, device)
        model, architecture = repaired_model(seed, 'compact-resnet')
        model[-1] = torch.nn.Linear(model[-1].in_features, len(CLASSES))
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        loader = DataLoader(dataset, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        best_score, best_epoch, state, history = -1.0, 0, None, []
        start = time.perf_counter()
        for epoch in range(1, 81):
            model.train()
            total_loss = 0.0
            for inputs, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs.to(device)), targets.to(device))
                if not torch.isfinite(loss).item():
                    raise ValueError('Loss CNN tidak finite.')
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(inputs)
            metrics = measure(labels[val], _cnn_scores(model, x[val], device))
            history.append({'epoch': epoch, 'loss': total_loss / len(dataset), **metrics})
            if metrics['macro_f1'] > best_score:
                best_score, best_epoch = metrics['macro_f1'], epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 5 == 0:
                print(f'CNN seed {seed} epoch {epoch}: val F1 {metrics["macro_f1"]:.4f}', flush=True)
            if epoch - best_epoch >= 16:
                break
        filename = f'resnet-seed-{seed}.pt'
        torch.save({'state_dict': state, 'mean': mean.reshape(-1).tolist(),
                    'std': std.reshape(-1).tolist(), 'labels': list(CLASSES),
                    'feature_config': FEATURES.to_dict(), 'architecture': architecture}, output / filename)
        model.load_state_dict(state)
        probabilities[seed] = _cnn_scores(model, x[test], device)
        result.append({'seed': seed, 'best_epoch': best_epoch, 'epochs': len(history),
                       'training_seconds': time.perf_counter() - start, 'artifact': filename,
                       'validation': history[best_epoch - 1], 'history': history,
                       'test': measure(labels[test], probabilities[seed])})
        write_json(output / f'resnet-seed-{seed}.json', result[-1])
        print('CNN', seed, result[-1]['test'], flush=True)
    chosen = max(result, key=lambda row: row['validation']['macro_f1'])
    return {'selected_seed': chosen['seed'], 'epochs': chosen['epochs'], 'best_epoch': chosen['best_epoch'],
            'validation': chosen['validation'], 'test': chosen['test'], 'seeds': result}, probabilities[chosen['seed']]


def train():
    from ketik_audio.devices import select_device

    device, hardware = select_device('gpu')
    print(f"GPU dikunci pada {hardware['name']}", flush=True)
    prepared, output = RUN / 'prepared', RUN / 'models'
    fresh_directory(output)
    with np.load(prepared / 'features.npz', allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    labels, fold = data['labels'], data['fold']
    masks = {role: fold == role for role in ROLES}
    results, scores = classical(data['mfcc'], labels, masks, output)
    write_json(output / 'classical.json', results)
    results['logmel-resnet'], scores['logmel-resnet'] = residual_cnn(data['mel'], labels, masks, output, device)
    test = masks['test']
    for name, score in scores.items():
        predicted = score.argmax(1)
        results[name]['confusion'] = confusion_matrix(labels[test], predicted, labels=LABEL_IDS).tolist()
        results[name]['by_prompt'] = {}
        for prompt in ('sentence', 'topup'):
            keep = data['prompt'][test] == prompt
            results[name]['by_prompt'][prompt] = {
                'n': int(keep.sum()), 'correct': int(np.sum(predicted[keep] == labels[test][keep])),
                'accuracy': float(np.mean(predicted[keep] == labels[test][keep])) if keep.any() else None,
            }
        np.save(output / f'{name}-test-scores.npy', score)
    with (output / 'predictions.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['event_seq', 'block', 'prompt', 'true', *scores])
        predictions = [score.argmax(1) for score in scores.values()]
        for i, (seq, block, prompt, label) in enumerate(zip(data['seq'][test], data['blocks'][test],
                                                          data['prompt'][test], labels[test])):
            writer.writerow([seq, block, prompt, CLASSES[label], *(CLASSES[p[i]] for p in predictions)])
    prior = int(np.bincount(labels[masks['train']]).argmax())
    write_json(output / 'results.json', {
        'labels': list(CLASSES), 'hardware': hardware, 'dataset_sha256': file_hash(prepared / 'dataset.json'),
        'features_sha256': file_hash(prepared / 'features.npz'),
        'chance_top1': 1 / len(CLASSES), 'train_majority_key': CLASSES[prior],
        'majority_top1': float(np.mean(labels[test] == prior)), 'test_count': int(test.sum()),
        'artifacts': {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()},
        'models': results,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train'))
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source)
    else:
        train()
