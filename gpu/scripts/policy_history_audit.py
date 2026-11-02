"""Read-only, deterministic audit of captured GPU inference and native history.

No simulator, Dota server, training, or weight modification is launched here.
Single-group substitutions are offline interventions, not closed-loop proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sf_duel.model import Policy
from sf_duel.native_data import examples
from sf_duel.spec import UNIT_FIELDS, SKILL_FIELDS, ITEM_FIELDS, GLOBAL_FIELDS, Act

LAYOUT = {
    'heroes': (2, 96), 'creeps': (256, 96), 'towers': (4, 96),
    'trees': (256, 8), 'projectiles': (512, 12), 'skills': (4, 12),
    'items': (8, 16), 'enemy_items': (8, 16), 'terrain': (4, 8, 8),
    'global_features': (6,), 'previous': (30,), 'hero_valid': (2,),
    'creep_valid': (256,), 'creep_ally': (256,), 'tree_valid': (256,),
    'projectile_valid': (512,), 'action_mask': (11,), 'buy_mask': (7,),
    'item_mask': (8,), 'attack_mask': (519,), 'tree_mask': (519,),
    'tp_mask': (519,), 'self_mask': (519,),
}
WIDTH = sum(int(np.prod(shape)) for shape in LAYOUT.values())
assert WIDTH == 37324
ACTION_COLUMNS = ('action', 'buy', 'item', 'target', 'x', 'y')
TARGET_MASKS = ('attack_mask', 'tree_mask', 'tp_mask', 'self_mask')


def is_boolean(key):
    return key.endswith(('_mask', '_valid')) or key == 'creep_ally'


def decode_packed(raw):
    if raw.ndim != 2 or raw.shape[1] != WIDTH:
        raise ValueError('Invalid packed observation shape')
    offset = 0
    result = {}
    for key, shape in LAYOUT.items():
        size = int(np.prod(shape))
        result[key] = raw[:, offset:offset + size].reshape(len(raw), *shape)
        offset += size
    return result


def canonical_native(arrays):
    """Remap variable native rows/target masks to the fixed GPU ABI."""
    n, t = len(arrays['creeps']), len(arrays['trees'])
    if n > 256 or t > 256 or len(arrays['projectiles']) > 512:
        raise ValueError('Native frame exceeds GPU capacities')
    result = {}
    for key, shape in LAYOUT.items():
        value = arrays[key]
        out = np.zeros(shape, dtype=np.float16)
        if key in TARGET_MASKS:
            out[:2] = value[:2]
            out[2:2 + n] = value[2:2 + n]
            out[258:262] = value[2 + n:6 + n]
            out[262:262 + t] = value[6 + n:6 + n + t]
            out[-1] = value[-1]
        elif value.shape == shape:
            out[:] = value
        else:
            out[:len(value)] = value
        result[key] = out
    return result


def tensor_batch(arrays, start=0, end=None):
    return {key: torch.from_numpy(np.asarray(value[start:end],
            dtype=np.bool_ if is_boolean(key) else np.float32).copy())
            for key, value in arrays.items()}


def comparison(actual, expected, atol=5e-4, rtol=5e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError('Comparison shape mismatch')
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    different = actual.astype('<f4').view('<u4') != expected.astype('<f4').view('<u4')
    close = np.isclose(actual, expected, atol=atol, rtol=rtol)
    return {'bit_equal': not bool(different.any()),
            'different_scalars': int(different.sum()), 'scalars': int(actual.size),
            'max_absolute_error': float(delta.max(initial=0)),
            'within_tolerance': bool(close.all()),
            'outside_tolerance': int((~close).sum())}


def expected_memory_chain(after, resets):
    before = np.zeros_like(after)
    before[1:] = after[:-1]
    # GPU reset is previous_state * (1-reset), including IEEE signed zero.
    return before * (1 - resets.astype(np.float32))[:, None]


def greedy_summary(logits, mask):
    masked = torch.as_tensor(np.asarray(logits, np.float32).copy()).masked_fill(
        ~torch.as_tensor(np.asarray(mask, np.bool_).copy()), -torch.inf)
    top = masked.topk(2, dim=-1).values
    return {'action': int(masked.argmax(-1)),
            'probabilities': masked.softmax(-1).tolist(),
            'top_two_logit_margin': float(top[0] - top[1])}


def replay_gpu(model, arrays, aux, columns):
    def col(name):
        lo, hi = columns[name]
        return aux[:, lo:hi]
    fused_parts, step_h, step_c, all_actions = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(aux), 32):
            end = min(start + 32, len(aux))
            b = tensor_batch(arrays, start, end)
            fused, keys = model.encode_entities(b)
            before = tuple(torch.from_numpy(col(k)[start:end].copy()).unsqueeze(0)
                           for k in ('before_h', 'before_c'))
            _, (h, c) = model.memory(fused.unsqueeze(1), before)
            captured_h = torch.from_numpy(col('after_h')[start:end].copy())
            chosen, *_ = model.heads(b, captured_h, keys,
                model.value_head(captured_h).squeeze(-1), None,
                deterministic=True, need_entropy=False)
            fused_parts.append(fused.numpy())
            step_h.append(h[0].numpy()); step_c.append(c[0].numpy())
            all_actions.append(np.stack([chosen[k].numpy() for k in ACTION_COLUMNS], 1))
        fused = np.concatenate(fused_parts)
        hs, cs = [], []
        state = model.initial_state(1)
        for i, value in enumerate(fused):
            if col('reset')[i, 0] > .5:
                state = model.initial_state(1)
            _, state = model.memory(torch.from_numpy(value.copy()).reshape(1, 1, 256), state)
            hs.append(state[0][0, 0].numpy()); cs.append(state[1][0, 0].numpy())
        hs, cs = np.stack(hs), np.stack(cs)
        logits = model.action_head(torch.from_numpy(hs)).numpy()
        head_logits = model.action_head(torch.from_numpy(col('after_h').copy())).numpy()
    metrics = {name: comparison(actual, expected) for name, actual, expected in [
        ('encoder', col('fused'), fused), ('one_step_h', col('after_h'), np.concatenate(step_h)),
        ('one_step_c', col('after_c'), np.concatenate(step_c)),
        ('full_history_h', col('after_h'), hs), ('full_history_c', col('after_c'), cs),
        ('action_head_given_gpu_h', col('action_logits_unmasked'), head_logits),
        ('full_history_logits', col('action_logits_unmasked'), logits)]}
    resets = col('reset')[:, 0] > .5
    chain = {}
    for before, after in [('before_h', 'after_h'), ('before_c', 'after_c')]:
        expected = expected_memory_chain(col(after), resets)
        chain[before] = comparison(col(before), expected, 0, 0)
    actions = col('actions').astype(np.int64)
    predicted = np.concatenate(all_actions)
    primary = np.where(np.asarray(arrays['action_mask'], bool), logits, -np.inf).argmax(-1)
    mismatch = np.flatnonzero(primary != actions[:, 0])
    report = {'frames': len(aux), 'atol': 5e-4, 'rtol': 5e-4,
              'numerics': metrics, 'reset_frames': np.flatnonzero(resets).tolist(),
              'memory_chain': chain,
              'greedy_actions_given_gpu_h': {key: int((predicted[:, i] != actions[:, i]).sum())
                                            for i, key in enumerate(ACTION_COLUMNS)},
              'full_history_primary_action_mismatches': int(len(mismatch)),
              'first_action_mismatch_frame': int(mismatch[0]) if len(mismatch) else None,
              'selected_primary_action_counts': np.bincount(actions[:, 0], minlength=11).tolist()}
    report['passed'] = (all(m['within_tolerance'] for m in metrics.values())
                        and all(m['bit_equal'] for m in chain.values())
                        and np.array_equal(predicted, actions) and not len(mismatch))
    return report, fused, hs, cs, logits


def native_history(model, run, team, until):
    selected = []
    for row in examples(run):
        if row['team'] != team:
            continue
        if row['encoded'].snapshot['time'] > until:
            break
        selected.append(row)
    if not selected:
        raise ValueError('Empty native history')
    frames = [canonical_native(row['encoded'].arrays) for row in selected]
    arrays = {key: np.stack([frame[key] for frame in frames]) for key in LAYOUT}
    fused_parts = []
    with torch.inference_mode():
        for start in range(0, len(selected), 32):
            fused, _ = model.encode_entities(tensor_batch(arrays, start, start + 32))
            fused_parts.append(fused)
        fused = torch.cat(fused_parts)
        state = model.initial_state(1)
        logits, before = [], []
        for i, row in enumerate(selected):
            if (row['sequence'] - 1) % 128 == 0:
                state = model.initial_state(1)
            before.append(tuple(v[0, 0].numpy().copy() for v in state))
            _, state = model.memory(fused[i:i+1, None], state)
            logits.append(model.action_head(state[0][0]).numpy()[0])
    logits = np.stack(logits)
    actions = np.asarray([row['action']['action'] for row in selected])
    predicted = np.where(arrays['action_mask'].astype(bool), logits, -np.inf).argmax(-1)
    mismatches = np.flatnonzero(actions != predicted)
    times = np.asarray([row['encoded'].snapshot['time'] for row in selected])
    move = np.flatnonzero(np.isin(actions, [1, 2]))
    return arrays, before, logits, selected, times, {
        'frames': len(selected), 'primary_action_mismatches': int(len(mismatches)),
        'first_action_mismatch_sequence': int(selected[mismatches[0]]['sequence']) if len(mismatches) else None,
        'first_movement_index': int(move[0]) if len(move) else None,
        'first_movement_sequence': int(selected[move[0]]['sequence']) if len(move) else None,
        'first_movement_time': float(times[move[0]]) if len(move) else None,
        'scope': 'Recorded native observations and previous actual commands; own offline memory, not a new closed-loop game.'}


def input_diff(gpu, native):
    result = {}
    field_groups = {'heroes': UNIT_FIELDS, 'creeps': UNIT_FIELDS, 'towers': UNIT_FIELDS,
                    'skills': SKILL_FIELDS, 'items': ITEM_FIELDS,
                    'enemy_items': ITEM_FIELDS, 'global_features': GLOBAL_FIELDS}
    for key in LAYOUT:
        got, want = gpu[key], native[key]
        metric = comparison(got, want, 0, 0)
        fields = field_groups.get(key)
        if fields:
            metric['known_bit_mismatches'] = int((got[..., len(fields):] != want[..., len(fields):]).sum())
        indices = np.argwhere(got.astype('<f2').view('<u2') != want.astype('<f2').view('<u2'))
        delta = np.abs(got.astype(np.float32) - want.astype(np.float32))
        changes = []
        for index in sorted(indices.tolist(), key=lambda i: float(delta[tuple(i)]), reverse=True)[:12]:
            record = {'index': index, 'gpu': float(got[tuple(index)]), 'native': float(want[tuple(index)])}
            if fields:
                record['field'] = list(fields)[index[-1] % len(fields)]
                record['known'] = index[-1] >= len(fields)
            changes.append(record)
        metric['examples'] = changes
        result[key] = metric
    return result


def one_frame(model, frame, before):
    with torch.inference_mode():
        b = tensor_batch({key: value[None] for key, value in frame.items()})
        fused, _ = model.encode_entities(b)
        state = tuple(torch.as_tensor(value.copy()).reshape(1, 1, 256) for value in before)
        _, state = model.memory(fused[:, None], state)
        logits = model.action_head(state[0][0]).numpy()[0]
    return greedy_summary(logits, frame['action_mask'])


def phase_replay(model, arrays, reset_flags):
    """Reset timing is the only modified factor; recorded previous actions stay."""
    with torch.inference_mode():
        fused = torch.cat([model.encode_entities(tensor_batch(arrays, i, i + 32))[0]
                           for i in range(0, len(reset_flags), 32)])
        state = model.initial_state(1)
        hs = []
        for i, reset in enumerate(reset_flags):
            if reset:
                state = model.initial_state(1)
            _, state = model.memory(fused[i:i+1, None], state)
            hs.append(state[0][0, 0])
        return model.action_head(torch.stack(hs)).numpy()


def bundle_parity(checkpoint_state, bundle):
    from weights import rust_name
    manifest = json.loads((bundle / 'manifest.json').read_text())
    result = []
    for name, tensor in checkpoint_state.items():
        expected = tensor.numpy()
        if expected.ndim == 2 and not name.endswith('_embed.weight'):
            expected = expected.T
        entry = manifest['tensors'][rust_name(name)]
        path = bundle / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Weight bundle checksum mismatch')
        actual = np.fromfile(path, '<f4').reshape(entry['shape'])
        result.append(comparison(actual, expected, 0, 0))
    return {'tensors': len(result), 'parameters': sum(x['scalars'] for x in result),
            'bit_equal': all(x['bit_equal'] for x in result)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--native-run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); started = time.monotonic()
    torch.set_num_threads(2); torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    capture = Path(a.capture); meta = json.loads((capture / 'policy-trace.json').read_text())
    for filename, key in [('policy-inputs.f16', 'inputs_sha256'), ('policy-aux.f32', 'aux_sha256')]:
        if hashlib.sha256((capture / filename).read_bytes()).hexdigest() != meta[key]:
            raise ValueError('Capture checksum mismatch: ' + filename)
    raw = np.fromfile(capture / 'policy-inputs.f16', '<f2').reshape(meta['observation_shape'])
    arrays = decode_packed(raw)
    aux = np.fromfile(capture / 'policy-aux.f32', '<f4').reshape(meta['aux_shape'])
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if saved['gpu_pipeline']['recurrent_reset_interval'] != 128 or saved['gpu_pipeline']['observation_storage'] != 'f16':
        raise ValueError('Expected128/F16 checkpoint contract')
    model = Policy().eval(); model.load_state_dict(saved['state_dict'], strict=True)
    config = json.loads((capture / 'departure-config.json').read_text())['config']
    weights = bundle_parity(saved['state_dict'], Path(config['initial_checkpoint']))
    gpu_report, _, _, _, _ = replay_gpu(model, arrays, aux, meta['columns'])
    print(json.dumps({'gpu_replay': gpu_report}), flush=True)
    native, before, logits, rows, times, native_report = native_history(model, a.native_run, meta['team'], 3.)
    print(json.dumps({'native_replay': native_report}), flush=True)
    ni = native_report['first_movement_index']
    if ni is None:
        raise ValueError('No native movement in bounded prefix')
    gpu_times = aux[:, meta['columns']['game_tick'][0]] / 30
    # Include the bracketing pre-event frame, but use the post-event frame for
    # the main comparison: nearest-time alone can cross the first-wave boundary.
    gi = int(np.searchsorted(gpu_times, times[ni], side='left'))
    g = {key: value[gi] for key, value in arrays.items()}
    n = {key: value[ni] for key, value in native.items()}
    cols = meta['columns']
    g_before = tuple(aux[gi, slice(*cols[key])] for key in ('before_h', 'before_c'))
    native_before = before[ni]
    interventions = {}
    for key in LAYOUT:
        # Exactly one packed group replaced. Associated validity remains fixed.
        interventions[key] = one_frame(model, dict(g, **{key: n[key]}), g_before)
    semantic_interventions = {'creep_features_and_validity': one_frame(model,
        dict(g, **{key: n[key] for key in ('creeps', 'creep_valid', 'creep_ally')}), g_before)}
    # Match periodic reset phase to the other recording's observed clock. The
    # two decision grids do not coincide: report the rounding explicitly.
    native_reset_indices = [i for i, row in enumerate(rows) if (row['sequence'] - 1) % 128 == 0]
    native_reset_times = times[native_reset_indices]
    mapped = np.searchsorted(gpu_times, native_reset_times, side='left')
    valid = (native_reset_times >= gpu_times[0]) & (mapped < len(aux))
    mapped = mapped[valid]
    phase_flags = np.zeros(len(aux), bool); phase_flags[0] = True; phase_flags[mapped] = True
    changed_phase_logits = phase_replay(model, arrays, phase_flags)
    phase_actions = np.where(arrays['action_mask'].astype(bool), changed_phase_logits, -np.inf).argmax(-1)
    phase_movement = np.flatnonzero(np.isin(phase_actions, [1, 2]))
    # A complete native reset block ending at the first MOVE. Before that
    # command, Radiant's recorded primary commands are all CONTINUE.
    block_start = ni - (rows[ni]['sequence'] - 1) % 128
    block_native = {key: value[block_start:ni+1].copy() for key, value in native.items()}
    aligned_indices = np.searchsorted(gpu_times, times[block_start:ni+1], side='left')
    block_gpu = {key: value[aligned_indices].copy() for key, value in arrays.items()}
    block_resets = np.zeros(ni - block_start + 1, bool); block_resets[0] = True
    factors = {key: (key,) for key in LAYOUT}
    factors.update(creep_observations=('creeps', 'creep_valid', 'creep_ally'),
                   tree_observations=('trees', 'tree_valid'),
                   projectile_observations=('projectiles', 'projectile_valid'))
    block_results = {}
    for factor, group_keys in factors.items():
        altered = dict(block_native, **{key: block_gpu[key] for key in group_keys})
        last_logits = phase_replay(model, altered, block_resets)[-1]
        block_results[factor] = greedy_summary(last_logits, altered['action_mask'][-1])
    # Narrower scalar hypotheses: preserve known bits and every other field.
    for factor, group, field in [('hero_attack_cycle_zero', 'heroes', 'attack_cycle'),
                                 ('tower_attack_cycle_zero', 'towers', 'attack_cycle')]:
        altered = dict(block_native); altered[group] = block_native[group].copy()
        altered[group][..., list(UNIT_FIELDS).index(field)] = 0
        block_results[factor] = greedy_summary(phase_replay(model, altered, block_resets)[-1],
                                               altered['action_mask'][-1])
    terrain_results = {}
    terrain_changed = block_native['terrain'] != block_gpu['terrain']
    for channel in range(4):
        altered = dict(block_native); altered['terrain'] = block_native['terrain'].copy()
        altered['terrain'][:, channel] = block_gpu['terrain'][:, channel]
        terrain_results['channel_' + str(channel)] = greedy_summary(
            phase_replay(model, altered, block_resets)[-1], n['action_mask'])
    for channel, y, x in np.argwhere(terrain_changed.any(0)):
        altered = dict(block_native); altered['terrain'] = block_native['terrain'].copy()
        altered['terrain'][:, channel, y, x] = block_gpu['terrain'][:, channel, y, x]
        terrain_results[f'cell_{channel}_{y}_{x}'] = {
            **greedy_summary(phase_replay(model, altered, block_resets)[-1], n['action_mask']),
            'changed_frames': int(terrain_changed[:, channel, y, x].sum()),
            'world_position_at_endpoint': [float(rows[ni]['encoded'].snapshot['self'][axis] + (cell - 3.5)*256)
                                           for axis, cell in [('x', x), ('y', y)]]}
    counts = lambda frame: {key: int(frame[key].sum()) for key in ('creep_valid', 'tree_valid', 'projectile_valid')}
    creep_positions = lambda frame: (frame['creeps'][frame['creep_valid'].astype(bool), :2].astype(float) * 8192).tolist()
    report = {'capture': str(capture), 'native_run': a.native_run,
        'checkpoint': a.checkpoint, 'checkpoint_sha256': hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        'gpu_capture_sha256': {'inputs': meta['inputs_sha256'], 'aux': meta['aux_sha256']},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checkpoint_vs_actual_gpu_weight_bundle': weights,
        'gpu_math_and_actions': gpu_report, 'native_offline_replay': native_report,
        'first_native_movement_comparison': {
            'gpu_frame': gi, 'native_sequence': rows[ni]['sequence'],
            'gpu_time': float(gpu_times[gi]), 'native_time': float(times[ni]),
            'gpu_prior_frame_time': float(gpu_times[gi-1]),
            'gpu_prior_frame_counts': counts({key: value[gi-1] for key, value in arrays.items()}),
            'native_memory_age_decisions': int((rows[ni]['sequence'] - 1) % 128),
            'gpu_memory_age_decisions': int(gi - np.flatnonzero(aux[:gi+1, cols['reset'][0]] > .5)[-1]),
            'time_alignment_error_seconds': float(gpu_times[gi] - times[ni]),
            'gpu_counts': counts(g), 'native_counts': counts(n),
            'gpu_creep_positions_after_fp16': creep_positions(g), 'native_creep_positions_after_fp16': creep_positions(n),
            'group_differences': input_diff(g, n),
            'gpu_inputs_gpu_memory': one_frame(model, g, g_before),
            'native_inputs_native_memory': one_frame(model, n, native_before),
            'native_inputs_gpu_memory': one_frame(model, n, g_before),
            'gpu_inputs_native_memory': one_frame(model, g, native_before),
            'gpu_with_one_native_packed_group_fixed_gpu_memory': interventions,
            'gpu_with_one_native_semantic_group_fixed_gpu_memory': semantic_interventions,
            'interpretation': 'Nearest game-time snapshots, not identical states/commands. Variable entity rows can be permuted. Single-group swaps preserve other masks and memory; they measure immediate sensitivity only. They cannot by themselves prove a physical cause or a closed-loop fix.'},
        'gpu_recorded_inputs_native_reset_phase_only': {
            'reset_frames': np.flatnonzero(phase_flags).tolist(),
            'mapped_native_reset_times': native_reset_times[valid].tolist(),
            'round_up_errors_seconds': (gpu_times[mapped] - native_reset_times[valid]).tolist(),
            'first_movement_frame': int(phase_movement[0]) if len(phase_movement) else None,
            'first_movement_game_time': float(gpu_times[phase_movement[0]]) if len(phase_movement) else None,
            'primary_actions_changed': int((phase_actions != aux[:, cols['actions'][0]]).sum()),
            'first_native_movement_time_result': greedy_summary(changed_phase_logits[gi], g['action_mask']),
            'scope': 'Offline replay: only reset phase changed. FP16 inputs and previous actual commands fixed, including after a counterfactual MOVE. Not closed-loop behavior or proof of a fix.'},
        'native_last_reset_block_one_factor_from_gpu': {
            'native_sequence_range': [rows[block_start]['sequence'], rows[ni]['sequence']],
            'frames': len(block_resets), 'native_start_time': float(times[block_start]),
            'gpu_alignment_errors_seconds': (gpu_times[aligned_indices] - times[block_start:ni+1]).tolist(),
            'native_reference': greedy_summary(logits[ni], n['action_mask']),
            'native_inputs_native_reset_from_zero_recomputed': greedy_summary(
                phase_replay(model, block_native, block_resets)[-1], n['action_mask']),
            'all_gpu_inputs_native_reset_from_zero': greedy_summary(
                phase_replay(model, block_gpu, block_resets)[-1], g['action_mask']),
            'one_factor_results': block_results,
            'terrain_one_channel_or_cell_results': terrain_results,
            'scope': 'Each factor independently modifies the same full26-step recorded native reset block. Associated validity is bundled only in explicit semantic factors. Primary actions preceding the endpoint are recordedCONTINUE. Nearest followingGPU samples differ by <=one decision period; this is offline sensitivity, not identical-state dynamics or a closed-loop fix.'},
        'wall_seconds': time.monotonic() - started,
        'torch_version': torch.__version__, 'training_updates': 0}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp'); tmp.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n', encoding='utf-8'); tmp.replace(out)
    print(json.dumps({'output': str(out), 'wall_seconds': report['wall_seconds'], 'gpu_passed': gpu_report['passed']}), flush=True)


if __name__ == '__main__':
    main()
