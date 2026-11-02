local selected = {}
function GetBotNames()
    local names={}
    for index=1,5 do names[index]="SF_DUEL_"..GetTeam().."_"..index end
    return names
end
function Think()
    for _,team in ipairs({TEAM_RADIANT,TEAM_DIRE}) do
        for index,pid in ipairs(GetTeamPlayers(team)) do
            if IsPlayerBot(pid) and not selected[pid] then
                SelectHero(pid,index==1 and "npc_dota_hero_nevermore" or "npc_dota_hero_lina")
                selected[pid]=true
            end
        end
    end
end
function UpdateLaneAssignments()
    local lanes={}
    for _,pid in ipairs(GetTeamPlayers(GetTeam())) do lanes[pid]=LANE_MID end
    return lanes
end
