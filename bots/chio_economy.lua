local agent=require('bots/agent')
local M={}
function M.Think(bot,st)
 -- Wraith Band belongs to mode setup; infrastructure must not buy another.
 st.permanent=agent.owned(bot,'item_bracer')>0 and 2 or (agent.owned(bot,'item_boots')>0 and 1 or 0)
 st.start_bought=true;agent.infrastructure(bot,st)
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
 if purchase and bot:GetGold()>=agent.missing_cost(bot,purchase) and GetItemStockCount(purchase)~=0 then
  local _,ok=agent.order_item(bot,purchase)
  if ok and purchase=='item_tango' then st.chio_tango=true end
 end
end
return M
