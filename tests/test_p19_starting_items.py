from pathlib import Path
import pytest
from lupa.lua51 import LuaRuntime

ROOT=Path(__file__).resolve().parents[1]


def economy_runtime():
    lua=LuaRuntime()
    lua.execute('''
gold=600;inventory={};courier_items={};bought={};time=-89
PURCHASE_ITEM_SUCCESS=-1
function DotaTime()return time end
function GameTime()return time+100 end
function GetTeam()return 2 end
function GetItemStockCount(name)return name=='item_tango' and 10 or 0 end
costs={item_circlet=155,item_slippers=140,item_recipe_wraith_band=210,item_tango=90,item_boots=500,item_bracer=505}
function GetItemCost(name)return costs[name] end
function print()end
function item(name)return {GetName=function()return name end,GetCurrentCharges=function()return name=='item_tango' and 3 or 0 end}end
courier={GetTeam=function()return 2 end,GetItemInSlot=function(_,slot)return courier_items[slot]end}
function GetCourier()return courier end
function GetCourierState()return 0 end
package.preload['game/dkjson']=function()return {encode=function()return '{}'end}end
package.preload['bots/settings']=function()return {}end
bot={GetGold=function()return gold end,GetLevel=function()return 1 end,GetAbilityPoints=function()return 0 end,
 GetPlayerID=function()return 0 end,GetItemInSlot=function(_,slot)return inventory[slot]end,
 IsAlive=function()return true end,GetHealth=function()return 582 end,GetMaxHealth=function()return 582 end,
 GetMana=function()return 291 end,GetMaxMana=function()return 291 end,GetModifierByName=function()return -1 end,
 GetCurrentMovementSpeed=function()return 305 end}
function bot:ActionImmediate_PurchaseItem(name)
 assert(gold>=costs[name]);gold=gold-costs[name];bought[#bought+1]=name
 local slot=0;while inventory[slot] do slot=slot+1 end;inventory[slot]=item(name)
 if name=='item_recipe_wraith_band' then inventory={ [0]=item('item_wraith_band') } end
 return PURCHASE_ITEM_SUCCESS
end
state={}
''')
    agent=lua.execute((ROOT/'bots/agent.lua').read_text())
    lua.globals().agent_module=agent
    lua.execute("package.preload['bots/agent']=function()return agent_module end")
    economy=lua.execute((ROOT/'bots/chio_economy.lua').read_text())
    return lua,economy


def test_starting_band_is_paid_before_tango_and_not_duplicated():
    lua,economy=economy_runtime()
    economy.Think(lua.globals().bot,lua.globals().state)
    assert lua.globals().gold==95
    assert list(lua.globals().bought.values())==['item_circlet','item_slippers','item_recipe_wraith_band']
    for _ in range(5):economy.Think(lua.globals().bot,lua.globals().state)
    assert lua.globals().gold==5
    assert list(lua.globals().bought.values())==['item_circlet','item_slippers','item_recipe_wraith_band','item_tango']


@pytest.mark.parametrize('location',['inventory[0]','inventory[9]','courier_items[0]'])
def test_existing_band_in_inventory_stash_or_courier_is_not_rebought(location):
    lua,economy=economy_runtime()
    lua.execute(location+"=item('item_wraith_band');gold=95")
    economy.Think(lua.globals().bot,lua.globals().state)
    assert lua.globals().state.start_bought
    assert list(lua.globals().bought.values())==['item_tango']
    assert lua.globals().gold==5


def test_setup_does_not_grant_items_or_touch_hero_gold():
    lua=LuaRuntime()
    lua.execute('''
Entities={FindAllByClassname=function()error('Setup must not grant hero items')end}
package.preload['bots/p17_bridge']=function()return {}end
package.preload['bots/play_model_events']=function()return {start=function()end}end
package.preload['game/dkjson']=function()return {}end
function ListenToGameEvent()end
''')
    lua.execute((ROOT/'gpu/bots/p17_setup.lua').read_text())
    lua.execute((ROOT/'gpu/bots/p17_setup.lua').read_text())
