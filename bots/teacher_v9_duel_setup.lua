-- One manual lobby setup operation in the server VM, separate from bot policy.
local M={}
function M.run(human_team)
 assert(human_team==2 or human_team==3,"Expected human team 2 or 3")
 local teacher_team=5-human_team
 local human_count,keep=0,nil
 local remove={}
 for pid=0,23 do
  if PlayerResource:IsValidPlayerID(pid) then
   local hero=PlayerResource:GetSelectedHeroEntity(pid)
   if PlayerResource:IsFakeClient(pid) then
    local name=PlayerResource:GetPlayerName(pid) or ""
    assert(string.find(name,"SF_DUEL_",1,true)==1,"Unexpected bot; no players were removed")
    local prefix="SF_DUEL_"..teacher_team.."_1"
    if string.find(name,prefix,1,true)==1 then
     assert(not keep and hero and hero:GetUnitName()=="npc_dota_hero_nevermore" and hero:GetTeamNumber()==teacher_team,"Teacher SF is not ready")
     keep=pid
    else remove[#remove+1]=pid end
   else
    human_count=human_count+1
    assert(hero and hero:GetTeamNumber()==human_team and hero:GetUnitName()=="npc_dota_hero_nevermore","Pick human SF and enter the game first")
   end
  end
 end
 assert(human_count==1 and keep,"Expected one human SF and one teacher SF; no players were removed")
 -- The same reset/disconnect operations are used by Valve's hero demo removal.
 for _,pid in ipairs(remove) do GameRules:ResetPlayer(pid);DisconnectClient(pid,true) end
 print("SF_V9_DUEL_READY|human_team="..human_team.."|teacher_player="..keep.."|removed="..#remove)
 return {human_team=human_team,teacher_player=keep,removed=#remove}
end
return M
