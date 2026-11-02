"""Read-only behavioral diagnostics from completed native match recordings."""
import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sf_duel.native_data import records, ITEM_IDS


def audit(path, team):
    commands = {}; acknowledgements = {}; requested = Counter(); effects = Counter()
    cooldowns = {}; previous_buffs = set(); recent = deque(); pending = None
    retreat_times = []; last_retreat = -100.; observations = 0
    item_names = {v: k for k, v in ITEM_IDS.items()}
    buffs = {'modifier_flask_healing', 'modifier_tango_heal', 'modifier_clarity_potion'}
    for row in records(path):
        if row.HasField('execution') and row.execution.team == team:
            e = row.execution
            acknowledgements[e.sequence] = (e.accepted, e.reason)
        if not row.HasField('decision') or row.decision.team != team:
            continue
        d = row.decision; state = row.observation; now = state.dota_time
        own = next(u for u in state.units if u.name == 'npc_dota_hero_nevermore' and u.team_id == team)
        enemy = next((u for u in state.units if u.name == 'npc_dota_hero_nevermore' and u.team_id != team and u.is_alive), None)
        observations += 1
        if d.action == 10:
            commands[d.sequence] = d.item; requested[d.item] += 1
        current_buffs = {m.name for m in own.modifiers} & buffs
        effects.update(current_buffs - previous_buffs); previous_buffs = current_buffs
        for item in own.items:
            if item.ability_id not in item_names:
                continue
            previous = cooldowns.get(item.handle, item.cooldown_remaining)
            if item.cooldown_remaining > previous + 1:
                effects['cooldown_started:' + item_names[item.ability_id]] += 1
            cooldowns[item.handle] = item.cooldown_remaining
        # A transparent retreat proxy, not a claim about tactical intent:
        # lose >=2% max HP within 1s while an enemy is within1200, then
        # issue movement and move >=256 away from its ORIGINAL position in3s.
        while recent and recent[0][0] < now - 1:
            recent.popleft()
        if pending:
            pending['move'] |= d.action in (1, 2)
            distance = math.hypot(own.location.x-pending['x'], own.location.y-pending['y'])
            if now > pending['time'] + 3:
                pending = None
            elif pending['move'] and distance >= pending['distance'] + 256:
                retreat_times.append(float(now)); last_retreat = now; pending = None
        if pending is None and now >= 0 and now-last_retreat >= 5 and enemy is not None:
            distance = math.hypot(own.location.x-enemy.location.x, own.location.y-enemy.location.y)
            # Compare only equal max HP, so level-ups/stat changes aren't damage.
            high = max((hp for _, hp, maximum in recent if maximum == own.health_max), default=own.health)
            if distance <= 1200 and high-own.health >= .02*own.health_max:
                pending = dict(time=now, x=enemy.location.x, y=enemy.location.y,
                               distance=distance, move=d.action in (1, 2))
        recent.append((now, own.health, own.health_max))
    accepted = Counter(); rejected = Counter(); missing = Counter()
    for sequence, name in commands.items():
        ack = acknowledgements.get(sequence)
        if ack is None:
            missing[name] += 1
        elif ack[0]:
            accepted[name] += 1
        else:
            rejected[name + ':' + ack[1]] += 1
    return dict(team=team, observations=observations, requested_items=dict(requested),
                accepted_item_commands=dict(accepted), rejected_item_commands=dict(rejected),
                unacknowledged_item_commands=dict(missing), observed_item_effects=dict(effects),
                retreat_proxy_times=retreat_times,
                limitations='ACK means command issued, not completed use. Buff/cooldown starts are observed effects. Retreat proxy is geometric, not proof of tactical intent; damage source and final killer are not inferred.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('recording'); parser.add_argument('--team', type=int, required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.recording, args.team), indent=2))
