"""GGD v3 timing ABI. Diagnostic fields are never populated by Dota."""
import numpy as np

# Append-only extension to ggd-v2; each row is values followed by known flags.
TIMING_FIELDS = {
    'attack_animation': 1., 'since_attack': 3., 'ready_in': 3.,
    'oracle_ready_in': 3., 'oracle_windup_in': 3.,
    'oracle_hit_in': 3., 'oracle_hp': 2000., 'oracle_damage_min': 200.,
    'oracle_damage_max': 200., 'oracle_death_in': 3., 'oracle_point_carry': 1.,
}
HONEST_FIELDS = 3


def timing_rows(rows):
    from .observations import features
    # A live snapshot cannot inject diagnostic values, even if it contains keys.
    return np.stack([features({k: r.get(k) for k in list(TIMING_FIELDS)[:HONEST_FIELDS]} if not r.get('creep_siege') else {}, TIMING_FIELDS)
                     for r in rows]) if rows else np.zeros((0, 2*len(TIMING_FIELDS)), np.float32)


def ablate(arrays, honest=True, oracle=False):
    """Identical masks for collection-derived training and simulator inference."""
    from .spec import UNIT_FIELDS, COMBAT_UNIT_FIELDS
    result = dict(arrays)
    fields = len(TIMING_FIELDS)
    at = 2*len(COMBAT_UNIT_FIELDS)
    for group, extra in [('heroes','hero_combat'), ('creeps','creep_combat'), ('towers','tower_combat')]:
        x = arrays[extra].copy()
        for begin,end,enabled in [(0,HONEST_FIELDS,honest),(HONEST_FIELDS,fields,oracle)]:
            if not enabled:
                x[...,at+begin:at+end] = 0
                x[...,at+fields+begin:at+fields+end] = 0
        result[extra] = x
        if not honest:
            x = arrays[group].copy()
            cycle = list(UNIT_FIELDS).index('attack_cycle')
            health = list(UNIT_FIELDS).index('health')
            x[...,cycle] = 0
            x[...,len(UNIT_FIELDS)+cycle] = x[...,len(UNIT_FIELDS)+health]
            result[group] = x
    return result
