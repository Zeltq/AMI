local json=require('game/dkjson')
local selected=nil
local before=nil
local done=false
local sent=false
local cut_time=nil
local function sample(p)
 local rows={}
 for y=-2,2 do for x=-2,2 do
  local q=Vector(p.x+x*32,p.y+y*32,p.z)
  rows[#rows+1]={dx=x*32,dy=y*32,passable=IsLocationPassable(q),height=GetHeightLevel(q)}
 end end
 return rows
end
function Think()
 local ok,err=pcall(function()
  local b=GetBot()
  if done or not b or GetTeam()~=TEAM_RADIANT or b:GetUnitName()~='npc_dota_hero_nevermore' then return end
  if GetGameState()~=GAME_STATE_PRE_GAME and GetGameState()~=GAME_STATE_GAME_IN_PROGRESS then return end
  local slot=b:FindItemSlot('item_tango')
  if slot<0 then b:ActionImmediate_PurchaseItem('item_tango');return end
  if not selected then
   local best=100000
   for _,t in pairs(b:GetNearbyTrees(1600)) do
    local p=GetTreeLocation(t);local d=GetUnitToLocationDistance(b,p)
    if IsLocationPassable(p) and d<best then best=d;selected=t end
   end
   if not selected then error('no nearby tree') end
   local p=GetTreeLocation(selected);before={id=selected,x=p.x,y=p.y,z=p.z,grid=sample(p)}
   print('SF_GPU_TREE_BEFORE|'..json.encode(before))
  end
  if not sent then b:Action_UseAbilityOnTree(b:GetItemInSlot(slot),selected);sent=true end
  if b:HasModifier('modifier_tango_heal') then
   if not cut_time then cut_time=DotaTime() end
   if DotaTime()<cut_time+.5 then return end
   local exists=false
   for _,t in pairs(b:GetNearbyTrees(1600)) do if t==selected then exists=true end end
   local p=Vector(before.x,before.y,before.z)
   print('SF_GPU_TREE_AFTER|'..json.encode({id=selected,tree_still_nearby=exists,grid=sample(p)}));done=true
  end
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err)) end
end
