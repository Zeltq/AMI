local json=require('game/dkjson')
local cursor=0
local started=false
local last_stats=-1000
local function unit(u)
 local p=u:GetLocation()
 return {name=u:GetUnitName(),team=u:GetTeam(),x=p.x,y=p.y,z=p.z,
  hp=u:GetHealth(),max_hp=u:GetMaxHealth(),mana=u:GetMana(),max_mana=u:GetMaxMana(),
  damage=u:GetAttackDamage(),range=u:GetAttackRange(),interval=u:GetSecondsPerAttack(),
  attack_point=u:GetAttackPoint(),projectile_speed=u:GetAttackProjectileSpeed(),
  speed=u:GetCurrentMovementSpeed(),armor=u:GetArmor(),magic_resist=u:GetMagicResist(),
  hp_regen=u:GetHealthRegen(),mana_regen=u:GetManaRegen(),level=u:GetLevel()}
end
local function work()
 local bot=GetBot()
 if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
 local phase=GetGameState()
 if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 if not started then
  started=true
  print('SF_GPU_HERO|'..json.encode(unit(bot)))
  if GetTeam()==TEAM_RADIANT then
   local trees={}
   for _,tree in pairs(GetAllTrees()) do
    local p=type(tree)=='number' and GetTreeLocation(tree) or tree
    trees[#trees+1]={x=p.x,y=p.y,z=p.z}
    if #trees==64 then print('SF_GPU_TREES|'..json.encode(trees));trees={} end
   end
   if #trees>0 then print('SF_GPU_TREES|'..json.encode(trees)) end
  end
   for _,team in ipairs({GetTeam()}) do
    for _,index in ipairs({TOWER_MID_1,TOWER_MID_2}) do
     local tower=GetTower(team,index)
     if tower then print('SF_GPU_TOWER|'..json.encode(unit(tower))) end
    end
    local p=GetShopLocation(team,SHOP_HOME)
    print('SF_GPU_BASE|'..json.encode({team=team,x=p.x,y=p.y,z=p.z}))
   end
 end
 if GetTeam()==TEAM_RADIANT and cursor<512*512 then
  local rows={};local first=cursor
  for i=1,128 do
   if cursor>=512*512 then break end
   local x=-8192+(cursor%512+.5)*32
   local y=-8192+(math.floor(cursor/512)+.5)*32
   local p=Vector(x,y,0)
   rows[#rows+1]={GetHeightLevel(p),IsLocationPassable(p) and 1 or 0}
   cursor=cursor+1
  end
  print('SF_GPU_GRID|'..json.encode({first=first,rows=rows}))
  if cursor==512*512 then print('SF_GPU_MAP_DONE') end
 end
 if DotaTime()>=last_stats+1 then
  last_stats=DotaTime()
  for _,u in pairs(GetUnitList(UNIT_LIST_ALLIED_CREEPS)) do
   if string.find(u:GetUnitName(),'npc_dota_creep_') or string.find(u:GetUnitName(),'siege') then
    print('SF_GPU_CREEP|'..json.encode(unit(u)))
   end
  end
 end
end
function Think()
 local ok,err=pcall(work)
 if not ok then print('SF_GPU_ERROR|'..tostring(err)) end
end
