"""Versioned observation and action contract, shared by inference and datasets."""
from enum import IntEnum

SPEC_VERSION = 1
SPELLS = ('nevermore_shadowraze1', 'nevermore_shadowraze2',
          'nevermore_shadowraze3', 'nevermore_requiem')
ITEMS = ('item_wraith_band', 'item_boots', 'item_bracer', 'item_tpscroll',
         'item_enchanted_mango', 'item_flask', 'item_tango', 'item_clarity')
BUYS = ('none', 'next_permanent', 'item_tpscroll', 'item_enchanted_mango',
        'item_flask', 'item_tango', 'item_clarity')
TOWERS = ('npc_dota_goodguys_tower1_mid', 'npc_dota_goodguys_tower2_mid',
          'npc_dota_badguys_tower1_mid', 'npc_dota_badguys_tower2_mid')

class Act(IntEnum):
    CONTINUE = 0
    MOVE = 1
    MOVE_DIRECT = 2
    ATTACK = 3
    STOP = 4
    HOLD = 5
    RAZE_NEAR = 6
    RAZE_MID = 7
    RAZE_FAR = 8
    REQUIEM = 9
    ITEM = 10

# Values absent from a snapshot get a separate known=0 feature, not an invented
# measurement. Enemy private fields are removed BEFORE this representation.
UNIT_FIELDS = {
    'x': 8192., 'y': 8192., 'z': 512., 'dx': 2048., 'dy': 2048.,
    'facing_cos': 1., 'facing_sin': 1., 'health': 2000., 'health_max': 2000.,
    'health_fraction': 1., 'mana': 1000., 'mana_max': 1000., 'mana_fraction': 1.,
    'health_regen': 20., 'mana_regen': 10., 'move_speed': 600.,
    'attack_damage': 200., 'attack_range': 1000., 'attack_interval': 3.,
    'attack_point': 1., 'projectile_speed': 1500., 'armor': 20.,
    'magic_resist': 1., 'level': 15., 'gold': 2000., 'xp_to_next': 2000.,
    'last_hits': 150., 'denies': 50., 'souls': 30.,
    'is_attacking': 1., 'is_using_ability': 1., 'is_casting_ability': 1.,
    'is_channeling': 1., 'fear': 1., 'fear_remaining': 3.,
    'raze_stacks': 5., 'raze_remaining': 6.,
    'flask': 1., 'clarity': 1., 'tango': 1.,
    'attack_target_self': 1., 'attack_target_enemy': 1.,
    'attack_cycle': 1., 'active_spell': 4., 'team_relative': 1.,
    'creep_melee': 1., 'creep_ranged': 1., 'creep_siege': 1.,
}
SKILL_FIELDS = {'level': 4., 'cooldown': 120., 'mana_cost': 200.,
                'cast_point': 2., 'castable': 1., 'channel_time': 3.}
ITEM_FIELDS = {'inventory': 10., 'backpack': 10., 'stash': 10., 'courier': 10.,
               'cooldown': 120., 'castable': 1., 'mana_cost': 200., 'slot': 20.}
TREE_FIELDS = {'dx': 2048., 'dy': 2048., 'z': 512., 'distance': 2048.}
PROJECTILE_FIELDS = {'dx': 2048., 'dy': 2048., 'speed': 1500.,
                     'is_attack': 1., 'caster_enemy': 1., 'target_self': 1.}
GLOBAL_FIELDS = {'time': 600., 'next_permanent': 2., 'side': 1.}
PREVIOUS_FIELDS = ('action', 'x', 'y', 'target', 'item', 'buy')
ENEMY_PRIVATE = {'gold', 'xp', 'xp_to_next', 'last_hits', 'denies', 'souls', 'fear_remaining',
                 'raze_remaining', 'active_spell'}
COORDINATES = tuple(range(-1024, 1025, 16))

def blank_action():
    return {'action': 0, 'x': 64, 'y': 64, 'target': 0, 'item': 0, 'buy': 0}
