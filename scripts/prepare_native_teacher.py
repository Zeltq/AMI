"""Build the small executor independently of Open Hyper AI runtime files."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def prepare(destination):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
 for name in ('agent.lua','hero_selection.lua','native_executor.lua','native_model_observation.lua'):shutil.copy2(ROOT/'bots'/name,destination/name)
 callback="local bridge=require('bots/native_executor')\nfunction Think() local ok,err=pcall(bridge.Think);if not ok then print('SF_NATIVE_ERROR|'..tostring(err)) end end\n"
 (destination/'bot_generic.lua').write_text(callback)
 (destination/'item_purchase_generic.lua').write_text(callback.replace('function Think()', 'function ItemPurchaseThink()'))
 (destination/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 for team in (2,3):(destination/f'native_commands_{team}.lua').write_text('return {}\n')
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.lua')}
 (destination/'ADAPTER.json').write_text(json.dumps({'controller':'native-go-experimental','hashes':hashes},indent=2))
 return destination
