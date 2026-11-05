-- Local Host setup: engine player names may be empty for Local Dev Script bots.
local M={}
function M.run(human_team)
 assert(human_team==2 or human_team==3,'Expected human team 2 or 3')
 local model_team=5-human_team
 local human_count,keep=0,nil
 local remove={}
 for pid=0,23 do
  if PlayerResource:IsValidPlayerID(pid) then
   local team=PlayerResource:GetTeam(pid)
   if team==2 or team==3 then
    local hero=PlayerResource:GetSelectedHeroEntity(pid)
    local name=hero and not hero:IsNull() and hero:GetUnitName() or nil
    if PlayerResource:IsFakeClient(pid) then
     if team==model_team and name=='npc_dota_hero_nevermore' then
      assert(not keep,'More than one opposing SF; no bots removed')
      keep=pid
     else
      assert(name==nil or name=='npc_dota_hero_lina','Unexpected extra bot hero; no bots removed')
      remove[#remove+1]=pid
     end
    else
     human_count=human_count+1
     assert(team==human_team and name=='npc_dota_hero_nevermore','Pick human SF on the configured side and wait for spawn')
    end
   end
  end
 end
 assert(human_count==1 and keep,'Expected one human SF and one opposing bot SF; no bots removed')
 assert(type(GameRules.ResetPlayer)=='function' and type(DisconnectClient)=='function','Server cleanup API unavailable; no bots removed')
 for _,pid in ipairs(remove) do GameRules:ResetPlayer(pid);DisconnectClient(pid,true) end
 print('SF_PLAY_COMPOSITION|human_team='..human_team..'|model_player='..keep..'|removed='..#remove)
 return {human_team=human_team,model_player=keep,removed=#remove}
end
return M
