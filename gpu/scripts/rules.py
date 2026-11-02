"""Parse installed KeyValues into a provenance-preserving JSON rule manifest."""
import argparse
import json
from pathlib import Path
import re

def parse(text):
    # Strip comments outside strings; quoted values, braces and #base are the
    # complete syntax used by the selected installed NPC rule files.
    tokens=re.findall(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|[{}]|[^\s{}"]+',text)
    tokens=[t[1:-1] if t.startswith('"') else t for t in tokens if not t.startswith('//')]
    position=0
    def block(nested=False):
        nonlocal position
        result={}
        while position<len(tokens):
            key=tokens[position];position+=1
            if key=='}':return result
            value=tokens[position];position+=1
            if value=='{':value=block(True)
            if key in result and isinstance(result[key],dict) and isinstance(value,dict):result[key].update(value)
            else:result[key]=value
        if nested:raise ValueError('Unclosed KeyValues block')
        return result
    return block()

def main():
    p=argparse.ArgumentParser();p.add_argument('calibration');a=p.parse_args();root=Path(a.calibration)
    hero=parse((root/'rules/npc_dota_hero_nevermore.txt').read_text(encoding='utf-8-sig'))
    units=parse((root/'rules/npc_units.txt').read_text(encoding='utf-8-sig'))
    heroes=parse((root/'rules/npc_heroes.txt').read_text(encoding='utf-8-sig'))
    items=parse((root/'rules/items.txt').read_text(encoding='utf-8-sig'))
    abilities=parse((root/'rules/npc_abilities.txt').read_text(encoding='utf-8-sig'))['DOTAAbilities']
    rules={'hero':hero,'hero_defaults':heroes,'units':{k:v for k,v in next(iter(units.values())).items() if k in ('dota_fountain','npc_dota_creep_goodguys_melee','npc_dota_creep_goodguys_ranged','npc_dota_creep_badguys_melee','npc_dota_creep_badguys_ranged','npc_dota_goodguys_siege','npc_dota_badguys_siege') or 'tower' in k},
        'items':{k:v for k,v in next(iter(items.values())).items() if k in ('item_wraith_band','item_boots','item_bracer','item_tpscroll','item_enchanted_mango','item_flask','item_tango','item_clarity')},
        'combat':{k:abilities[k] for k in ('creep_siege','creep_irresolute','creep_piercing','backdoor_protection')}}
    (root/'rules.json').write_text(json.dumps(rules,indent=2))
    print(json.dumps({'hero_roots':list(hero),'hero_keys':list(next(iter(hero.values()))),'hero_defaults':list(heroes)[:5],'unit_count':len(rules['units'])}))
if __name__=='__main__':main()
