from pathlib import Path
import torch
from lupa.lua51 import LuaRuntime
from sf_duel.chio_cpu import Session

ROOT=Path(__file__).resolve().parents[1]


def test_navigation_queue_strict_boundary_and_latch():
    lua=LuaRuntime(unpack_returned_tuples=True)
    lua.execute('''messages={};moves={};x=-6000;y=-6000;tick=0
package.preload['game/dkjson']=function()return {encode=function()return '{}' end}end
function print()end
function DotaTime()return tick/30 end
function Vector(x,y,z)return {x=x,y=y,z=z}end
TOWER_MID_1=0
function GetTower()return {IsNull=function()return false end,GetLocation=function()return Vector(-1544,-1408,0)end}end
bot={GetLocation=function()return Vector(x,y,0)end,GetTeam=function()return 2 end,
 Action_MoveToLocation=function(_,p)moves[#moves+1]={tick=tick,x=p.x,y=p.y}end}
''')
    nav=lua.execute((ROOT/'bots/chio_navigation.lua').read_text());state=nav.new(True);g=lua.globals()
    for tick in range(8):g.tick=tick;nav.update(state,g.bot,tick)
    assert len(g.moves)==1 and g.moves[1].tick==7
    p=g.moves[1];length=(1544**2+1408**2)**.5
    assert abs(p.x-(-1544+1544/length*350))<1e-9
    g.x=1200;g.y=0;g.tick=12
    assert nav.update(state,g.bot,12)==(False,False) # abs(x-y)==1200 fails strict criterion
    g.x=1000;g.y=1000;g.tick=18
    assert nav.update(state,g.bot,18)==(True,True)
    g.x=6000;g.y=6000
    for tick in range(19,60):g.tick=tick;ready,handoff=nav.update(state,g.bot,tick);assert ready and not handoff
    assert state.handoffs==1
    assert max(row.tick for row in g.moves.values())==19 # last queued pre-handoff observation at12


def test_handoff_reset_preserves_global_periodic_schedule():
    class Fake:
        def decide(self,b,state,deterministic):
            self.state_before=state
            return {'action':torch.tensor([1])},None,None,torch.tensor([0.]),('memory',)
    model=Fake();session=Session(model);resets=[]
    for i in range(260):
        _,_,reset=session.decide({'x':torch.tensor([1.])},i*6,'episode',handoff=i==137)
        if reset:resets.append(i)
        if i==137:assert model.state_before is None
    assert resets==[0,128,137,256]
