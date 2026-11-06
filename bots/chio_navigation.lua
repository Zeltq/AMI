-- Shared P16 base phase. Only public own/tower positions; no policy features.
local json=require('game/dkjson')
local M={}
function M.new(enabled)
 return {enabled=enabled,ready=not enabled,pending={},handoffs=0}
end
function M.update(state,bot,tick)
 if not state.enabled then return true,false end
 if not state.start_tick then state.start_tick=tick end
 local handoff=false
 if tick%6==0 then
  local p=bot:GetLocation()
  if not state.ready and p.x*p.x+p.y*p.y<2000*2000 and math.abs(p.x-p.y)<1200 then
   state.ready=true;state.handoff_tick=tick;state.handoffs=state.handoffs+1;handoff=true
   print('SF_CHIO_HANDOFF|'..json.encode({team=bot:GetTeam(),tick=tick,time=DotaTime(),
    elapsed_seconds=(tick-state.start_tick)/30,handoffs=state.handoffs,x=p.x,y=p.y,previous_action=1,reset_required=true}))
  end
  if not state.ready then
   local tower=GetTower(bot:GetTeam(),TOWER_MID_1)
   assert(tower and not tower:IsNull(),'own mid T1 unavailable for navigation')
   local p=tower:GetLocation();local length=math.max(1,math.sqrt(p.x*p.x+p.y*p.y))
   state.pending[#state.pending+1]={observed_tick=tick,execute_tick=tick+7,x=p.x-p.x/length*350,y=p.y-p.y/length*350}
  end
 end
 while state.pending[1] and state.pending[1].execute_tick<=tick do
  local d=table.remove(state.pending,1);local accepted=d.execute_tick==tick
  if accepted then bot:Action_MoveToLocation(Vector(d.x,d.y,0)) end
  print('SF_CHIO_NAV_ACK|'..json.encode({team=bot:GetTeam(),observed_tick=d.observed_tick,execute_tick=d.execute_tick,
   tick=tick,accepted=accepted,x=d.x,y=d.y,ready=state.ready,order='Action_MoveToLocation'}))
 end
 return state.ready,handoff
end
return M
