-- Mode setup only: these fields never enter the model transport.
for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do
 if not h:HasItemInInventory('item_wraith_band') then h:AddItemByName('item_wraith_band') end
end
require('bots/p17_bridge')
require('bots/play_model_events').start()
