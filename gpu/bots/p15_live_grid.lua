-- Read-only startup export, loaded through the existing server Lua bridge.
local ok,err=pcall(function()
 local json=require('game/dkjson')
 local hero=Entities:FindAllByClassname('npc_dota_hero_nevermore')[1]
 assert(hero,'SF must be spawned before export')
 local metadata={game_time=GameRules:GetDOTATime(false,true),height_api=type(GetGroundHeight),grid_mapping={}}
 for _,x in ipairs({-128,-64,-32,-16,0,16,32,48,64,128})do
  local success,value=pcall(function()local index=GridNav:WorldToGridPosX(x);return{x=x,index=index,center=GridNav:GridPosToWorldCenterX(index)}end)
  metadata.grid_mapping[#metadata.grid_mapping+1]=success and value or {x=x,error=tostring(value)}
 end
 print('SF_P15_GRID_META|'..json.encode(metadata))
 local trees=GridNav:GetAllTreesAroundPoint(Vector(0,0,0),30000,true)
 assert(type(trees)=='table' and #trees>100,'tree enumeration unavailable')
 for i,t in ipairs(trees)do local p=t:GetAbsOrigin();local standing=true
  if type(t.IsStanding)=='function' then standing=t:IsStanding()end
  print('SF_P15_GRID_TREE|'..json.encode({id=t:entindex(),x=p.x,y=p.y,z=p.z,alive=standing}))
 end
 local seen={}
 for _,class in ipairs({'npc_dota_tower','npc_dota_barracks','npc_dota_fort','npc_dota_building','npc_dota_filler','ent_dota_fountain'})do
  for _,u in ipairs(Entities:FindAllByClassname(class))do if not seen[u:entindex()]then seen[u:entindex()]=true;local p=u:GetAbsOrigin()
   local row={id=u:entindex(),class=class,x=p.x,y=p.y,z=p.z}
   for _,name in ipairs({'GetUnitName','GetTeamNumber','GetHullRadius','GetHealth','GetMaxHealth','GetBoundingMins','GetBoundingMaxs'})do
    local success,v=pcall(function()return u[name](u)end)
    if success then row[name]=(name=='GetBoundingMins' or name=='GetBoundingMaxs')and{x=v.x,y=v.y,z=v.z}or v end
   end
   print('SF_P15_GRID_BUILDING|'..json.encode(row))
  end end
 end
 local step=64;local row=0
 hero:SetContextThink('p15_live_grid',function()
  local success,failure=pcall(function()
   local side=16384/step
   for chunk=0,side/64-1 do local values={}
    for i=0,63 do local x=chunk*64+i;local p=Vector(-8192+(x+.5)*step,-8192+(row+.5)*step,0)
     local z=GetGroundHeight(p,hero)
     values[#values+1]=GridNav:IsTraversable(p)and 1 or 0
     values[#values+1]=GridNav:IsBlocked(p)and 1 or 0
     values[#values+1]=z
    end
    print('SF_P15_GRID_ROW|'..json.encode({step=step,row=row,chunk=chunk,values=values}))
   end
  end)
  if not success then print('SF_P15_GRID_ERROR|'..tostring(failure));return nil end
  row=row+1
  if row==16384/step then if step==64 then step=32;row=0 else print('SF_P15_GRID_DONE|true');return nil end end
  return .03
 end,0)
end)
if not ok then print('SF_P15_GRID_ERROR|'..tostring(err))end
