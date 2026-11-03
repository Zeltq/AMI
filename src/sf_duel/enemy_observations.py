"""ggd-v4: visible casts, observed onset clocks, levels and raze debuffs.

Enemy cooldown getters return the unavailable sentinel -1 in build 6970.
Clocks below measure observed cast ONSETS, including subsequently cancelled casts.
They are not cooldown estimates. Missing visibility never advances event memory.
"""
import numpy as np

FIELDS = {**{f'cast_{i}': 1. for i in range(5)}, 'cast_phase': 1.,
          **{f'since_cast_{i}': 60. for i in range(1,5)},
          **{f'spell_level_{i}': 4. for i in range(1,5)},
          'raze_stacks': 16., 'raze_remaining': 10.}

def hero_rows(snapshot):
    from .observations import features
    return np.stack([features((snapshot.get(k) or {}).get('v4', {})
                             if (snapshot.get(k) or {}).get('visible',k=='self') else {}, FIELDS)
                     for k in ('self','enemy')])

class CastMemory:
    """Shared causal algorithm for native Dota and event-parity fixtures."""
    def __init__(self):
        self.units = {}

    def observe(self, key, time, spell, visible, levels=(), stacks=None, remaining=None):
        state = self.units.setdefault(key, dict(previous=None, start=None, last={}, time=None))
        if not visible:
            state.update(previous=None,start=None,time=None)
            return {}
        result = {}
        if state['time'] is not None and time-state['time']>.2:state.update(previous=None,start=None)
        if spell is not None:
            result.update({f'cast_{i}':int(spell==i) for i in range(5)})
            if spell != state['previous']:
                state['start'] = time if spell and state['previous'] is not None else None
                if state['start'] is not None:state['last'][spell]=time
            if spell==0:result['cast_phase']=0.
            elif state['start'] is not None:result['cast_phase']=min(1.,max(0.,time-state['start'])/(1.67 if spell==4 else .55))
            state['previous']=spell
        else:state.update(previous=None,start=None)
        state['time']=time
        for i,t in state['last'].items():result[f'since_cast_{i}']=min(60.,max(0.,time-t))
        for i,level in enumerate(levels,1):
            if level is not None:result[f'spell_level_{i}']=level
        if stacks is not None:result['raze_stacks']=stacks
        if remaining is not None:result['raze_remaining']=remaining
        return result
