local agent=require('bots/agent')
local json=require('game/dkjson')
local M={}
local function trace(bot,st,row)
 row.time=DotaTime();row.team=GetTeam();row.player=bot:GetPlayerID();row.gold=bot:GetGold()
 row.inventory={};row.courier_inventory={}
 for slot=0,20 do local it=bot:GetItemInSlot(slot);if it then row.inventory[#row.inventory+1]={slot=slot,name=it:GetName(),charges=it:GetCurrentCharges()} end end
 local courier=GetCourier(0)
 if courier then
  row.courier_team=courier:GetTeam();row.courier_state=GetCourierState(courier)
  for slot=0,8 do local it=courier:GetItemInSlot(slot);if it then row.courier_inventory[#row.courier_inventory+1]={slot=slot,name=it:GetName(),charges=it:GetCurrentCharges()} end end
 end
 print('SF_CHIO_ECONOMY|'..json.encode(row))
end
function M.Think(bot,st)
 -- Buy the starting Band through the shared shop path before other purchases.
 st.permanent=agent.owned(bot,'item_bracer')>0 and 2 or (agent.owned(bot,'item_boots')>0 and 1 or 0)
 local starting=not st.start_bought;local before=bot:GetGold()
 agent.infrastructure(bot,st)
 if starting and bot:GetGold()<before then
  trace(bot,st,{kind='start_purchase',item='item_wraith_band',gold_before=before,cost=before-bot:GetGold(),success=agent.owned(bot,'item_wraith_band')>0})
 end
 if not st.start_bought then return end
 if not bot:IsAlive() then return end
 local tango=agent.owned(bot,'item_tango')
 if tango>0 then st.chio_tango=true end
 local purchase=nil
 if DotaTime()<0 and not st.chio_tango then purchase='item_tango'
 else
  local flask=agent.owned(bot,'item_flask');local mango=agent.owned(bot,'item_enchanted_mango')
  local missing=bot:GetMaxHealth()-bot:GetHealth();local low=bot:GetLevel()<5
  local healing=0
  for _,v in ipairs({{'modifier_flask_healing',30},{'modifier_tango_heal',7}}) do
   local index=bot:GetModifierByName(v[1])
   if index>=0 then healing=healing+math.max(0,bot:GetModifierRemainingDuration(index))*v[2] end
  end
  local need_flask=(flask<1 and tango==0) or (flask<(low and 2 or 3) and missing>400*flask+112*tango+healing-100)
  local need_mango=(missing<100 or not need_flask) and math.min(bot:GetMaxMana(),low and 360 or 450)>110*mango+bot:GetMana() and mango<(low and 3 or 4)
  if need_flask then purchase='item_flask'
  elseif need_mango then purchase='item_enchanted_mango'
  elseif agent.owned(bot,'item_boots')==0 then purchase='item_boots'
  elseif agent.owned(bot,'item_bracer')==0 then purchase='item_bracer' end
 end
 local cost=purchase and agent.missing_cost(bot,purchase) or 0
 local stock=purchase and GetItemStockCount(purchase) or nil
 -- GetItemStockCount returns 0 for these unrestricted permanent items in Bot API.
 -- Their purchase result, including component failures, is authoritative.
 local permanent=purchase=='item_boots' or purchase=='item_bracer'
 if purchase and bot:GetGold()>=cost and (permanent or stock~=0) then
  local before=bot:GetGold();local results,ok,attempts=agent.order_item(bot,purchase)
  trace(bot,st,{kind='attempt',item=purchase,cost=cost,stock=stock,gold_before=before,api_results=results,attempts=attempts,success=ok})
  if ok and purchase=='item_tango' then st.chio_tango=true end
 elseif st.economy_trace_time==nil or GameTime()-st.economy_trace_time>=1 then
  st.economy_trace_time=GameTime()
  trace(bot,st,{kind='state',item=purchase,cost=cost,stock=stock,reason=not purchase and 'complete' or (bot:GetGold()<cost and 'gold' or 'stock_zero'),move_speed=bot:GetCurrentMovementSpeed()})
 end
end
return M
