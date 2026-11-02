"""Column encoding changes transport size, never the model observation contract."""
from .spec import UNIT_FIELDS

UNIT_COLUMNS=tuple(dict.fromkeys(('id','name','type','visible','team','attackable','tp','attack_target_id',*UNIT_FIELDS)))
PROBE_SKILL_COLUMNS=('slot','name','level')
INVENTORY_COLUMNS=('slot','name')


def unpack_snapshot(s):
    if s.get('codec')!='sf-columns-1':return s
    s.pop('codec')
    def row(values,columns):
        if len(values)!=len(columns):raise ValueError('Invalid telemetry column count')
        return {key:value for key,value in zip(columns,values) if value is not None}
    for key in ('self','enemy'):
        if key in s:s[key]=row(s[key],UNIT_COLUMNS)
    for key in ('creeps','towers'):
        s[key]=[row(value,UNIT_COLUMNS) for value in s[key]]
    s['probe']['skill_slots']=[row(value,PROBE_SKILL_COLUMNS) for value in s['probe']['skill_slots']]
    s['probe']['inventory']=[row(value,INVENTORY_COLUMNS) for value in s['probe']['inventory']]
    return s
