-- Observe scheduler callbacks during forced states; never issues orders.
local json=require('game/dkjson');local last={}
local function callback(name)
 local bot=GetBot();if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore'then return end
 local phase=GetGameState();if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5);if last[name]==tick then return end;last[name]=tick
 print('SF_GPU_BOT_CALLBACK|'..json.encode({name=name,team=GetTeam(),tick=tick,time=DotaTime(),fear=bot:HasModifier('modifier_nevermore_requiem_fear'),alive=bot:IsAlive()}))
end
function AbilityUsageThink()callback('ability')end
function AbilityLevelUpThink()callback('level')end
function CourierUsageThink()callback('courier')end
function ItemUsageThink()callback('item')end
function BuybackUsageThink()callback('buyback')end
