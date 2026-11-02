local json=require('game/dkjson')
local reported=0
function Think()
 local ok,err=pcall(function()
  local b=GetBot()
  if not b or b:GetUnitName()~='npc_dota_hero_nevermore' or reported==b:GetLevel() then return end
  if GetGameState()~=GAME_STATE_PRE_GAME and GetGameState()~=GAME_STATE_GAME_IN_PROGRESS then return end
  reported=b:GetLevel()
  print('SF_GPU_LEVEL|'..json.encode({team=GetTeam(),level=b:GetLevel(),xp_needed=b:GetXPNeededToLevel(),
   hp=b:GetMaxHealth(),mana=b:GetMaxMana(),damage=b:GetAttackDamage(),armor=b:GetArmor(),
   magic_resist=b:GetMagicResist(),interval=b:GetSecondsPerAttack(),point=b:GetAttackPoint()}))
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err)) end
end
