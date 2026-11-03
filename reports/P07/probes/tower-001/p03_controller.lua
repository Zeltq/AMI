local M={}
local json=require('game/dkjson')
local last=nil
function M.Think()
 local second=math.floor(DotaTime())
 if second~=last then last=second;print('SF_P03_HERO|'..json.encode({time=DotaTime(),team=GetTeam()})) end
end

return M
