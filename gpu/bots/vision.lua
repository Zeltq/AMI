-- Diagnostic only: stationary SF heroes, native clock and visibility samples.
local json=require('game/dkjson')
local agent=require('bots/agent')
local infrastructure={skill_index=1,permanent=0,seq=0}
local last_tick=-1
local last_sample=-10000
local atlas_rows={}
function Think()
 local ok,err=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
  local phase=GetGameState()
  if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last_tick then return end
  last_tick=tick
  agent.infrastructure(bot,infrastructure)
  local time=DotaTime()
  local route=require('bots/vision_route')
  if route.static_atlas and bot:GetCurrentVisionRange()==0 then
   local phase=time<0 and'pregame'or time<300 and'day'or'night'
   local y=atlas_rows[phase]or 0
   if y<128 then
    local start=GetTeam()==TEAM_RADIANT and -8192 or 4096
    local center,plus,minus={},{},{}
    for x=0,127 do
     local px=start+(x+.5)*32;local py=start+(y+.5)*32
     center[#center+1]=IsLocationVisible(Vector(px,py,0))and'1'or'0'
     plus[#plus+1]=IsLocationVisible(Vector(px+8,py+8,0))and'1'or'0'
     minus[#minus+1]=IsLocationVisible(Vector(px-8,py-8,0))and'1'or'0'
    end
    print('SF_GPU_ATLAS|'..json.encode({team=GetTeam(),phase=phase,row=y,start=start,step=32,
     center=table.concat(center),plus=table.concat(plus),minus=table.concat(minus),time=time,game_time=GameTime()}))
    atlas_rows[phase]=y+1
   end
  end
  if route.move and time>=-80 then
   local target=GetTeam()==TEAM_RADIANT and Vector(-600,-400,0)or Vector(500,400,0)
   if GetUnitToLocationDistance(bot,target)>16 then bot:Action_MoveToLocation(target)else bot:Action_ClearActions(true)end
  else bot:Action_ClearActions(true)end
  if time<last_sample+2 and not(time>=299 and time<=301) then return end
  last_sample=time
  local p=bot:GetLocation()
  local row={team=GetTeam(),tick=tick,time=time,game_time=GameTime(),x=p.x,y=p.y,z=p.z,grid={},trees={}}
  row.clock=GetTimeOfDay()
  row.global_api={}
  for _,name in ipairs({'GetGroundHeight','GetGroundPosition','GetWorldBounds'}) do row.global_api[name]=type(_G[name]) end
  if type(GetWorldBounds)=='function' then
   local success,value=pcall(function()return json.encode(GetWorldBounds())end)
   if success then row.world_bounds_json=value end
  end
  row.vision_api={}
  for _,name in ipairs({'GetCurrentVisionRange','GetDayTimeVisionRange','GetNightTimeVisionRange'}) do
   if type(bot[name])=='function' then
    local success,value=pcall(bot[name],bot)
    if success then row.vision_api[name]=value end
   end
  end
  row.sources={};row.source_errors=0
  if type(UNIT_LIST_ALLIES)=='number' then
   for _,u in ipairs(GetUnitList(UNIT_LIST_ALLIES))do
    local success,value=pcall(function()
     if u:IsAlive()and u:GetTeam()==GetTeam()then
      local radius=u:GetCurrentVisionRange();local q=u:GetLocation()
      if radius>0 then return{name=u:GetUnitName(),x=q.x,y=q.y,radius=radius}end
     end
    end)
    if success and value then row.sources[#row.sources+1]=value elseif not success then row.source_errors=row.source_errors+1 end
   end
  end
  for y=1,8 do for x=1,8 do
   local q=Vector(p.x+(x-4.5)*256,p.y+(y-4.5)*256,p.z)
   local cell={height=GetHeightLevel(q),visible=IsLocationVisible(q),passable=IsLocationPassable(q)}
   if type(GetGroundHeight)=='function' then local success,z=pcall(GetGroundHeight,q);if success then cell.ground_z=z end end
   row.grid[#row.grid+1]=cell
  end end
  for _,id in ipairs(bot:GetNearbyTrees(1600)) do
   local q=GetTreeLocation(id)
   row.trees[#row.trees+1]={id=id,x=q.x,y=q.y,z=q.z,visible=IsLocationVisible(q)}
  end
  print('SF_GPU_VISION|'..json.encode(row))
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err)) end
end
