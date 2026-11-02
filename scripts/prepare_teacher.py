"""Build an isolated, pinned OHA runtime; never changes the upstream snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.telemetry import UNIT_COLUMNS,PROBE_SKILL_COLUMNS,INVENTORY_COLUMNS
from sf_duel.live.bridge import literal
UPSTREAM=ROOT/'external/openhyperai'
TARGET=ROOT/'teacher_bots'

def prepare():
    if not (UPSTREAM/'UPSTREAM.json').exists():raise RuntimeError('Pinned upstream snapshot is missing')
    shutil.copytree(UPSTREAM/'bots',TARGET,dirs_exist_ok=True)
    (TARGET/'order_probe.lua').unlink(missing_ok=True)
    (TARGET/'team_desires.lua').unlink(missing_ok=True)
    shutil.copy2(UPSTREAM/'LICENSE',TARGET/'LICENSE')
    for name in ('agent.lua','teacher_monitor.lua','teacher_controller.lua','telemetry.lua','hero_selection.lua'):
        shutil.copy2(ROOT/'bots'/name,TARGET/name)
    (TARGET/'telemetry_schema.lua').write_text('return '+literal({'unit':UNIT_COLUMNS,
        'skills':PROBE_SKILL_COLUMNS,'inventory':INVENTORY_COLUMNS})+'\n')
    (TARGET/'telemetry_schema.json').write_text(json.dumps({'unit':UNIT_COLUMNS,
        'skills':PROBE_SKILL_COLUMNS,'inventory':INVENTORY_COLUMNS}))
    (TARGET/'item_purchase_generic.lua').write_text('''function ItemPurchaseThink()
    local phase=GetGameState()
    if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
    local bot=GetBot()
    if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
    local ok,err=pcall(function() require("bots/teacher_controller").Think() end)
    if not ok then print("SF_RULE_ERROR|"..require("game/dkjson").encode({reason=tostring(err),source="purchase_clock"})) end
end
''')
    disabled=('farm','rune','roshan','secret_shop','team_roam','push_tower_top','push_tower_bot',
              'defend_tower_top','defend_tower_bot','ward')
    for mode in disabled:
        (TARGET/f'mode_{mode}_generic.lua').write_text('function GetDesire() return 0 end\nfunction Think() end\n')
    path=TARGET/'ability_item_usage_generic.lua'
    source=path.read_text(encoding='utf-8')
    marker='return X';assert source.rstrip().endswith(marker)
    adapter='''
-- SF duel adapter: retain OHA combat/item decisions, replace only fixed rules.
local original_items=ItemUsageThink
-- Full bot Think owns the only decision clock; engine callbacks do nothing.
function AbilityUsageThink() end
function ItemUsageThink() end
function AbilityLevelUpThink() end
function CourierUsageThink() end
function BuybackUsageThink() end
X.AbilityUsageThink=AbilityUsageThink
X.ItemUsageThink=ItemUsageThink
X.AbilityLevelUpThink=AbilityLevelUpThink
X.BuybackUsageThink=BuybackUsageThink
X.DuelItemUsageThink=original_items
'''
    source=source[:source.rfind(marker)]+adapter+'\nreturn X\n'
    path.write_text(source,encoding='utf-8')
    (TARGET/'bot_generic.lua').write_text('''local duel=require("bots/teacher_controller")
function Think()
    local ok,err=pcall(duel.Think)
    if not ok then print("SF_RULE_ERROR|"..require("game/dkjson").encode({reason=tostring(err),source="teacher_controller"})) end
end
''')
    path=TARGET/'FunLib/aba_role.lua'
    source=path.read_text(encoding='utf-8')
    for function in ('GetPosition','GetRoleFromId'):
        marker='____exports.'+function+' = function(bot)'
        assert marker in source
        source=source.replace(marker,marker+'\n    if GetGameMode()==21 then bot.assignedRole=2;return 2 end',1)
    path.write_text(source,encoding='utf-8')
    for path in TARGET.rglob('*.lua'):
        source=path.read_text(encoding='utf-8')
        if 'debug.traceback()' in source:
            path.write_text(source.replace('debug.traceback()',
                '(debug and debug.traceback and debug.traceback() or "traceback unavailable in Bot API")'),encoding='utf-8')
    path=TARGET/'FunLib/aba_global_overrides.lua'
    source=path.read_text(encoding='utf-8')
    source='''-- Keep native getters for the observation collector before OHA guards.
if not _G.SF_NATIVE_BOT_API then
    _G.SF_NATIVE_BOT_API={}
    for name,fn in pairs(CDOTA_Bot_Script) do
        if type(fn)=="function" and not string.match(name,"^Action") then _G.SF_NATIVE_BOT_API[name]=fn end
    end
end
'''+source
    source=source.replace('function print(...)','function print(...)\n    local first=select(1,...)\n    if type(first)=="string" and string.sub(first,1,3)=="SF_" then orig_print(...);return end',1)
    source=source.replace('function GetTeamPlayers(nTeam, bypass)',
        'function GetTeamPlayers(nTeam, bypass)\n    if GetGameMode()==21 then local ids=orig_GetTeamPlayers(nTeam);return ids[1] and {ids[1]} or {} end',1)
    path.write_text(source,encoding='utf-8')
    for obsolete in TARGET.glob('_duel_original_mode_*_generic.lua'):obsolete.unlink()
    for path in list(TARGET.glob('mode_*_generic.lua')):
        source=path.read_text(encoding='utf-8')
        wrapper='''
local function checked(fn,name,fallback)
    return function(...)
        local ok,value=pcall(fn,...)
        if ok and string.find(name,"GetDesire",1,true) and type(value)~="number" then
            ok=false;value="GetDesire returned "..type(value)
        end
        if not ok then
            print("SF_RULE_ERROR|"..require("game/dkjson").encode({reason=tostring(value),source=name}))
            return fallback
        end
        return value
    end
end
if GetDesire then GetDesire=checked(GetDesire,"'''+path.name+''' GetDesire",0) end
if Think then Think=checked(Think,"'''+path.name+''' Think",nil) end
'''
        final=re.search(r'(?m)^return[^\n]*\s*$',source)
        if final:source=source[:final.start()]+wrapper+source[final.start():]
        else:source+=wrapper
        path.write_text(source,encoding='utf-8')
    path=TARGET/'mode_laning_generic.lua'
    source=path.read_text(encoding='utf-8').replace('if currentTime < 0 then return BOT_ACTION_DESIRE_NONE end',
        'if currentTime < 0 then return 0.45 end')
    path.write_text(source,encoding='utf-8')
    path=TARGET/'FunLib/override_generic/mode_laning_generic.lua'
    source=path.read_text(encoding='utf-8')
    marker='local nFurthestEnemyAttackRange = GetFurthestEnemyAttackRange(tEnemyHeroes)'
    assert marker in source
    source=source.replace(marker,marker+'\n\tif bot:GetUnitName()=="npc_dota_hero_nevermore" then nFurthestEnemyAttackRange=350 end',1)
    path.write_text(source,encoding='utf-8')
    # Never load server-side FretBots/Buff scripts. No added XP/gold/items/chat API.
    manifest=json.loads((UPSTREAM/'UPSTREAM.json').read_text())
    manifest.update(training_data_collection='explicit launch option',controller='full Lua at policy frequency; no native bot decisions',disabled_modes=list(disabled),
        rules='WB; choose Boots then Bracer; self consumables; fixed skills; no Frenzy/stats',
        hashes={p.relative_to(TARGET).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in TARGET.rglob('*.lua') if not p.name.startswith('response_') and p.name!='settings.lua'})
    (TARGET/'ADAPTER.json').write_text(json.dumps(manifest,indent=2))
    return TARGET

if __name__=='__main__':print(prepare())
