-- CPU port of gpu/src/scripted_teacher.rs v9. No Dota calls in this module.
-- Source SHA256: e9ed506fff0a3f51be809cb33fe567a78eb76cd7d8baa87811c0f0460e22c624
-- The live adapter supplies observed/inferred attack phases, not engine internals.
local M={version="gpu-teacher-v9-lua-1",
 source_sha256="e9ed506fff0a3f51be809cb33fe567a78eb76cd7d8baa87811c0f0460e22c624"}
local pi=math.pi
local atan2=math.atan2 or function(y,x) return math.atan(y,x) end
local function clamp(x,a,b) return math.max(a,math.min(b,x)) end
local function distance(a,b) local x,y=a.x-b.x,a.y-b.y;return math.sqrt(x*x+y*y) end
local function wrap(x) return (x+pi)%(2*pi)-pi end
local hulls={[0]=24,[1]=144,[2]=16,[3]=8,[4]=16}
local spread={[0]=3,[1]=2,[2]=2,[3]=2.5,[4]=5.5}
local prototypes={[0]=50,[1]=100,[2]=21,[3]=23,[4]=40}
local combat={1,.5,1,1,.5, 1,.7,1,1,.7, .75,.7,1,1,.7, .5,.35,1.5,1.5,.35, 1,2.5,1,1,1}
local function physical(s,raw,kind,q)
 local armor=q.armor or 0;local table_=s.combat or combat
 return math.ceil(raw*table_[kind*5+q.kind+1]*(1-.06*armor/(1+.06*math.abs(armor))))
end
local function reach(p,q) return p.range+(p.hull or hulls[p.kind])+(q.hull or hulls[q.kind]) end
local function flight(s,p,dist)
 if p.projectile_speed<=0 then return 0 end
 local length=dist
 if p.kind==0 then
  local x,y=s.muzzle_forward or 68.500274,s.muzzle_side or 150.694285
  length=math.sqrt((dist-x)^2+y*y)
 end
 return math.max(0,math.ceil((length-8)/p.projectile_speed*30)-1)/30
end
local function own_hit(s,p,q)
 local dist=distance(p,q);local range=reach(p,q);local rate=p.turn_rate or .9
 local approach=math.ceil(math.max(0,dist-range)/math.max(1,p.speed)*30)/30
 local angle=math.abs(wrap(atan2(q.y-p.y,q.x-p.x)-p.facing))
 local turn=math.ceil(math.max(0,angle-.2-rate*.4712389)/rate)/30
 local execute=s.time+(s.execute_ticks or 7)/30;local ready=p.ready or 0
 if (p.windup or 0)>s.time and execute>=p.windup then ready=math.max(ready,p.windup+p.interval-p.point) end
 local moving_turn=math.ceil(math.max(0,angle-2.3561945)/(rate*.4712389))/30
 local travel=approach>0 and math.max(turn,approach+moving_turn) or turn
 local start=math.max(execute+travel,ready)
 local point=math.max(0,math.ceil(p.point*30-(p.rounding_error or 0)-.0001))/30
 return start+point+flight(s,p,math.min(dist,range))
end
function M.new() return {target=nil,release=-1e20,hit=-1e20,item_until=-1e20,retreat=false,seen={}} end

function M.decide(s,m)
 local me=s.own;local now=s.time
 local a={action=0,buy=0,item=0,x=me.x,y=me.y,target_id=0}
 local units={};local by_id={}
 for _,u in ipairs(s.units) do
  if u.alive~=false and u.hp>0 then
   units[#units+1]=u;by_id[u.id]=u;m.seen[u.id]={damage=u.damage,kind=u.kind}
  end
 end
 table.sort(units,function(a_,b_)return a_.id<b_.id end)
 local function allowed(map,index) return map and map[index]==true end
 if s.dota_time<0 and (s.tango_count or 0)==0 and allowed(s.can_buy,5) then a.buy=5
 elseif me.max_hp-me.hp>200 and (s.flask_count or 0)==0 and not me.flask and allowed(s.can_buy,4) then a.buy=4
 elseif allowed(s.can_buy,1) then a.buy=1 end
 if not s.can_act or me.hp<=0 then return a,{forecasts={},candidates=0} end
 local threats=0
 for _,u in ipairs(units) do
  if u.kind~=0 and u.team~=me.team and (u.target==me.id or ((u.windup or 0)>now and u.windup_target==me.id)) then threats=threats+1 end
 end
 local fraction=me.hp/me.max_hp
 if fraction<.30 then m.retreat=true end
 if fraction>.56 and threats==0 then m.retreat=false end
 local source_death={}
 for _,u in ipairs(units) do source_death[u.id]=1e20 end
 local enemy=s.enemy
 if enemy and (enemy.windup or 0)>now then
  local q=by_id[enemy.windup_target]
  if q and q.kind>=2 and q.hp<=physical(s,enemy.damage-(enemy.spread or 3),0,q) then
   source_death[q.id]=enemy.windup+flight(s,enemy,distance(enemy,q))
  end
 end
 for _,shot in ipairs(s.shots or {}) do
  local q=by_id[shot.target];local caster=m.seen[shot.caster]
  if q and q.kind>=2 and caster and caster.kind==0 then
   local raw=caster.damage-3-4
   if raw>0 and q.hp<=physical(s,raw,0,q) then
    local eta=math.max(0,(distance(shot,q)-8)/math.max(1,shot.speed))
    source_death[q.id]=math.min(source_death[q.id],now+eta)
   end
  end
 end
 local anchor,closest=nil,1e20
 for _,q in ipairs(units) do
  if q.kind>=2 and q.team~=me.team then
   local d=distance(me,q);if d<closest then anchor=q;closest=d end
  end
 end
 local forecasts={};local best=nil;local score=1e20;local candidates=0
 local cx,cy,weight=0,0,0
 for _,q in ipairs(units) do
  local dist=distance(me,q)
  if q.kind>=2 and dist<2200 then
   if anchor and closest<2200 and distance(q,anchor)<750 then
    local w=1+2*(1-q.hp/q.max_hp);cx=cx+q.x*w;cy=cy+q.y*w;weight=weight+w
   end
   if dist<1000 then
    local hit=own_hit(s,me,q)
    if (me.windup or 0)>now and me.windup_target==q.id then hit=me.windup+flight(s,me,dist)
    elseif m.target==q.id and now<m.release then hit=m.hit end
    local delay=hit-now
    local minimum=physical(s,me.damage-(me.spread or 3),0,q)
    local maximum=physical(s,me.damage+(me.spread or 3),0,q)
    local damage,uncertainty={},{}
    for i=0,95 do damage[i]=0;uncertainty[i]=0 end
    local own_shot=false
    for _,shot in ipairs(s.shots or {}) do
     if shot.target==q.id then
      local tick=math.max(0,math.ceil((distance(shot,q)-8)/math.max(1,shot.speed)*30)-1)+1
      if tick<96 then
       local cache=m.seen[shot.caster];local kind=cache and cache.kind or shot.kind or 2
       local amount=cache and cache.damage or (s.prototypes or prototypes)[kind]
       damage[tick]=damage[tick]+physical(s,amount,kind,q)
       uncertainty[tick]=uncertainty[tick]+(kind==0 and 3 or 4)
       if shot.caster==me.id then own_shot=true end
      end
     end
    end
    for _,source in ipairs(units) do
     if source.id~=me.id and source.target==q.id then
      local walk=math.max(0,distance(source,q)-reach(source,q))/math.max(1,source.speed)
      local release=(source.windup or 0)>0 and source.windup or math.max(now+walk,source.ready or 0)+source.point
      local first=math.max(0,math.ceil((release-now+flight(s,source,distance(source,q)))*30))+(source.projectile_speed>0 and 1 or 0)
      local interval=math.max(1,source.interval*30);local amount=physical(s,source.damage,source.kind,q)
      local variance=(source.spread or spread[source.kind])^2/3
      for cycle=0,3 do
       local tick=math.ceil(first+cycle*interval);local launch=release+cycle*source.interval
       if tick<96 and launch<=(source_death[source.id] or 1e20)+1/30 and (source.kind~=0 or cycle==0) then
        damage[tick]=damage[tick]+amount;uncertainty[tick]=uncertainty[tick]+variance
       end
      end
     end
    end
    local hp,variance,sigma,predicted,deadline=q.hp,0,0,q.hp,96
    local landing=math.floor(delay*30+.5)
    for i=0,95 do
     hp=hp+(q.regen or 0)/30-damage[i];variance=variance+uncertainty[i]
     if i==landing then predicted=hp;sigma=math.sqrt(variance) end
     if hp<=0 and deadline==96 then deadline=i end
    end
    local viable=(q.team~=me.team or q.hp<=q.max_hp*.5) and q.attackable~=false and landing>=0 and landing<96 and predicted>.5 and predicted<=minimum and not own_shot
    local f={id=q.id,hit=hit,predicted_hp=predicted,damage_min=minimum,damage_max=maximum,deadline=deadline,viable=viable}
    forecasts[q.id]=f
    if viable then
     candidates=candidates+1
     local candidate=deadline+delay*.25+sigma*.02+(q.team==me.team and 6 or 0)
     if candidate<score then best=q;score=candidate end
    end
   end
  end
 end
 local current=m.target and by_id[m.target];local protected=false
 if current then
  protected=now<m.release or ((me.windup or 0)>now and me.windup_target==current.id)
  local f=forecasts[current.id];local release=(me.windup or 0)>now and me.windup or m.release-1/30
  if protected and f and f.predicted_hp<=0 and best and best.id~=current.id and now+8/30<release then protected=false end
 end
 local gx,gy=me.x,me.y
 if not protected then
  if best then
   local f=forecasts[best.id];a.action=3;a.target_id=best.id
   m.target=best.id;m.release=f.hit-flight(s,me,distance(me,best))+1/30;m.hit=f.hit
  else
   local sign=me.team==2 and -1 or 1
   gx=me.team==2 and -1050 or 100;gy=me.team==2 and -950 or 200
   if weight>0 then gx=cx/weight+sign*210;gy=cy/weight+sign*210 end
   local tower=s.enemy_tower;local tanks=0
   if tower then
    for _,u in ipairs(units) do if u.kind>=2 and u.team==me.team and distance(u,tower)<850 then tanks=tanks+1 end end
    local dx,dy=gx-tower.x,gy-tower.y;local d=math.max(1,math.sqrt(dx*dx+dy*dy))
    if tower.alive~=false and tanks<2 and d<1000 then gx=tower.x+dx/d*1000;gy=tower.y+dy/d*1000 end
   end
   if distance(me,{x=gx,y=gy})>90 then a.action=1
   elseif me.order==3 or me.order==1 or me.order==2 then a.action=4 end
   m.target=nil
  end
 end
 if now<m.item_until then a.action=0
 elseif not protected or fraction<.20 then
  if m.retreat and s.own_tower then
   local sign=me.team==2 and -1 or 1
   gx=s.own_tower.x+sign*220;gy=s.own_tower.y+sign*220
   a.action=distance(me,{x=gx,y=gy})>90 and 1 or 4;m.target=nil
  end
  if threats==0 and me.max_hp-me.hp>200 and not me.flask and allowed(s.can_item,5) then
   a.action=10;a.item=5;a.target_id=me.id;m.item_until=now+.4
  elseif me.max_hp-me.hp>100 and not me.tango and allowed(s.can_item,6) then
   local tree,nearest=nil,135
   for _,v in ipairs(s.trees or {}) do local d=distance(me,v);if d<nearest then tree=v;nearest=d end end
   if tree then a.action=10;a.item=6;a.target_id=tree.id;m.item_until=now+.4 end
  end
 end
 if a.action~=3 and not protected then m.target=nil end
 if a.action==1 then
  local dx,dy=gx-me.x,gy-me.y;local scale=math.max(1,math.abs(dx)/1024,math.abs(dy)/1024)
  a.x=me.x+(clamp(math.floor(64+dx/scale/16+.5),0,128)-64)*16
  a.y=me.y+(clamp(math.floor(64+dy/scale/16+.5),0,128)-64)*16
 end
 return a,{forecasts=forecasts,candidates=candidates,protected=protected,selected=best and best.id,threats=threats}
end
M.physical=physical;M.own_hit=own_hit;M.flight=flight
return M
