from pathlib import Path
from lupa.lua51 import LuaRuntime

ROOT=Path(__file__).resolve().parents[1]


def test_zero_stock_permanent_items_reach_purchase_api():
    lua=LuaRuntime();lua.execute('''
owned={item_tango=3,item_flask=1};bought={};gold=2000
function GetTeam()return 2 end
function DotaTime()return 100 end
function GameTime()return 100 end
function GetCourier()return nil end
function GetItemStockCount()return 0 end
function print()end
package.preload['game/dkjson']=function()return {encode=function()return '{}'end}end
package.preload['bots/agent']=function()return {
 owned=function(_,name)return owned[name] or 0 end,infrastructure=function()end,
 missing_cost=function()return 500 end,
 order_item=function(_,name)bought[#bought+1]=name;owned[name]=1;gold=gold-500;return {-1},true,{} end}end
bot={IsAlive=function()return true end,GetHealth=function()return 500 end,GetMaxHealth=function()return 500 end,
 GetMana=function()return 500 end,GetMaxMana=function()return 500 end,GetLevel=function()return 1 end,
 GetGold=function()return gold end,GetModifierByName=function()return -1 end,GetPlayerID=function()return 0 end,
 GetItemInSlot=function()return nil end,GetCurrentMovementSpeed=function()return 305 end}
state={}
''')
    e=lua.execute((ROOT/'bots/chio_economy.lua').read_text())
    e.Think(lua.globals().bot,lua.globals().state)
    e.Think(lua.globals().bot,lua.globals().state)
    assert list(lua.globals().bought.values())==['item_boots','item_bracer']


def test_economy_and_agent_lua_compile():
    lua=LuaRuntime()
    for f in ['bots/agent.lua','bots/chio_economy.lua']:
        lua.execute('assert(loadstring(...))',(ROOT/f).read_text())
