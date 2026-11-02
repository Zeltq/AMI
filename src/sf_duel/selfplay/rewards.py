"""Server-only reward state. Never feed this state to the actor encoder."""
import numpy as np

# Each row: last hits, denies, current HP, current max HP, T1 HP, T1 max HP.
def potential(radiant, dire):
    a, b = np.asarray(radiant, dtype=np.float64), np.asarray(dire, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 6 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Reward requires complete, finite server state for both players')
    for side in (a, b):
        if (side[..., (3, 5)] <= 0).any() or (side[..., (0, 1, 2, 4)] < 0).any():
            raise ValueError('Unknown/invalid HP or counters must not be replaced with zero')
        if (side[..., 2] > side[..., 3] + 1e-4).any() or (side[..., 4] > side[..., 5] + 1e-4).any():
            raise ValueError('HP exceeds current maximum')
    return (np.clip(.01*(a[..., 0]-b[..., 0]), -.30, .30)
            + np.clip(.003*(a[..., 1]-b[..., 1]), -.06, .06)
            + .10*(a[..., 2]/a[..., 3]-b[..., 2]/b[..., 3])
            + .20*(a[..., 4]/a[..., 5]-b[..., 4]/b[..., 5]))


def transition(phi, next_phi, result=0., terminal=False, gamma=1.):
    if gamma != 1.: raise ValueError('Approved objective is undiscounted')
    if not terminal and result != 0: raise ValueError('Result reward only belongs to true terminal')
    if result not in (-1, 0, 1): raise ValueError('Invalid game result')
    return result + (0. if terminal else next_phi) - phi


def gae(rewards, values, terminals, bootstrap=0., gamma=1., lam=1.):
    r, v, done = np.asarray(rewards, np.float64), np.asarray(values, np.float64), np.asarray(terminals, bool)
    if r.ndim != 1 or r.shape != v.shape or r.shape != done.shape or not len(r):
        raise ValueError('GAE requires a nonempty ordered trajectory')
    if not np.isfinite(r).all() or not np.isfinite(v).all() or not np.isfinite(bootstrap):
        raise ValueError('Nonfinite GAE input')
    if done[-1] and bootstrap != 0: raise ValueError('True terminal must not bootstrap')
    advantages = np.empty_like(r); carry = 0.; following = bootstrap
    for i in range(len(r)-1, -1, -1):
        alive = 1. - float(done[i])
        delta = r[i] + gamma * following * alive - v[i]
        carry = delta + gamma * lam * alive * carry
        advantages[i] = carry; following = v[i]
    return advantages.astype(np.float32), (advantages + v).astype(np.float32)


def server_state(world, team):
    hero = next((u for u in world.units if u.name == 'npc_dota_hero_nevermore' and u.team_id == team), None)
    tower_name = 'npc_dota_goodguys_tower1_mid' if team == 2 else 'npc_dota_badguys_tower1_mid'
    tower = next((u for u in world.units if u.name == tower_name and u.team_id == team), None)
    if hero is None or tower is None: raise ValueError('Missing authoritative own hero/T1 reward state')
    if any(not hero.HasField(k) for k in ('last_hits', 'denies', 'health', 'health_max')) or any(not tower.HasField(k) for k in ('health','health_max')):
        raise ValueError('Reward state field is unknown')
    return np.asarray([hero.last_hits, hero.denies, hero.health, hero.health_max, tower.health, tower.health_max], np.float32)
