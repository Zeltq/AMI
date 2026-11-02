"""Deterministic, observation-only SF teacher. Never called by neural policies."""
from collections import defaultdict,deque
import math
from .spec import *

def distance(a,b): return math.hypot(a['x']-b['x'],a['y']-b['y'])
def fraction(u,name): return u.get(name,0)/max(1,u.get(name+'_max',1))

class Teacher:
    VERSION='sf-script-6'
    def __init__(self):
        self.history=defaultdict(lambda:deque(maxlen=64))
        self.lock_until=-1e9
        self.move_until=-1e9
        self.attack_ready=-1e9
        self.retreating=False

    def decide(self,s):
        a=blank_action();me=s['self'];enemy=s.get('enemy');now=s['time'];team=s['team']
        creeps=s.get('creeps',[]);items=s['items'];skills=s['skills']
        refs=[me,enemy or {}]+creeps+s['towers']+s['trees']+[s['base']]
        def target(ref):
            a['target']=next(i for i,r in enumerate(refs) if r is ref)
            return ref
        def move(point,direct=False):
            a['action']=int(Act.MOVE_DIRECT if direct else Act.MOVE)
            a['x']=max(0,min(128,round((point['x']-me['x'])/16)+64))
            a['y']=max(0,min(128,round((point['y']-me['y'])/16)+64))
            return a,me
        def use(index,ref=None):
            a['action']=int(Act.ITEM);a['item']=index
            return a,target(ref or me)
        own_creeps=[c for c in creeps if c['team']==team]
        foe_creeps=[c for c in creeps if c['team']!=team]
        for c in creeps:
            hist=self.history[c['id']];hist.append((now,c['health']))
            while len(hist)>2 and now-hist[0][0]>1.5: hist.popleft()
        foe_distance=distance(me,enemy) if enemy else 1e9
        hp=fraction(me,'health');mp=fraction(me,'mana')
        missing_hp=me['health_max']-me['health'];missing_mp=me['mana_max']-me['mana']
        def count(index): return sum(items[index].get(k,0) or 0 for k in ('inventory','backpack','stash','courier'))
        # Shop choices use exactly the same availability mask as the network.
        if now<0 and count(6)==0 and s['legal']['buys'][5]:a['buy']=5
        elif missing_hp>250 and count(5)==0 and not me.get('flask') and s['legal']['buys'][4]: a['buy']=4
        elif missing_mp>120 and count(4)==0 and foe_distance<1100 and s['legal']['buys'][3]: a['buy']=3
        elif missing_mp>150 and count(7)==0 and s['legal']['buys'][6]: a['buy']=6
        elif missing_hp>140 and count(6)==0 and s['legal']['buys'][5]: a['buy']=5
        elif s['legal']['buys'][1]: a['buy']=1
        elif count(3)==0 and distance(me,s['base'])>2000 and hp<.45 and s['legal']['buys'][2]: a['buy']=2
        if me.get('fear') or not s['legal']['actions'][1]: return a,me
        if now<self.lock_until or me.get('is_channeling') or me.get('is_casting_ability'): return a,me
        if items[5].get('castable') and missing_hp>250 and not me.get('flask') and foe_distance>1000:
            self.lock_until=now+.15;return use(5)
        if items[4].get('castable') and me['mana']<90 and foe_distance<1000:
            self.lock_until=now+.15;return use(4)
        if items[7].get('castable') and missing_mp>150 and not me.get('clarity') and foe_distance>1000:
            self.lock_until=now+.15;return use(7)
        enemy_t1=s['towers'][2 if team==2 else 0]
        nearby_trees=[t for t in s['trees'] if distance(me,t)<700
            and distance(t,enemy_t1)>1000 and distance(t,s['base'])<distance(me,s['base'])+100]
        if items[6].get('castable') and missing_hp>100 and not me.get('tango') and nearby_trees:
            self.lock_until=now+.2;return use(6,min(nearby_trees,key=lambda t:distance(me,t)))
        own_t1=s['towers'][0 if team==2 else 2]
        if hp<.32:self.retreating=True
        if hp>.88:self.retreating=False
        if self.retreating:
            if items[3].get('castable') and foe_distance>1000 and distance(me,s['base'])>1800:
                self.lock_until=now+3.2;return use(3,s['base'])
            sign=-1 if team==2 else 1
            fountain={'x':s['base']['x']+sign*1000,'y':s['base']['y']+sign*1000}
            if distance(me,fountain)>150:return move(fountain)
            return a,me
        tanking=[c for c in own_creeps if distance(c,enemy_t1)<650]
        if enemy_t1.get('attack_target_self') or (distance(me,enemy_t1)<820 and len(tanking)<2):
            dx,dy=me['x']-enemy_t1['x'],me['y']-enemy_t1['y'];length=max(1,math.hypot(dx,dy))
            return move({'x':enemy_t1['x']+dx/length*1000,'y':enemy_t1['y']+dy/length*1000})
        if skills[3].get('castable') and skills[3].get('level',0)>0 and foe_distance<325 and me.get('souls',0)>=8 and hp>.35:
            a['action']=int(Act.REQUIEM);self.lock_until=now+1.8;return a,me
        # Last-hit/deny prediction uses observed HP history, never hidden future state.
        damage=me.get('attack_damage',40)
        reach=me.get('attack_range',525)
        if now>=self.attack_ready:
            candidates=[]
            for c in creeps:
                if not c.get('attackable') or distance(me,c)>reach+30: continue
                armor=c.get('armor',0)
                dealt=max(1,damage-3)*(1-.06*armor/(1+.06*abs(armor)))
                flight=distance(me,c)/max(1,me.get('projectile_speed',1200))
                delay=me.get('attack_point',.5)+flight+.08
                hist=self.history[c['id']]
                dps=max(0,(hist[0][1]-hist[-1][1])/max(.8,hist[-1][0]-hist[0][0]))
                incoming=sum(u.get('attack_damage',0)/max(.3,u.get('attack_interval',1.5))
                    for u in creeps if u.get('attack_target_id')==c['id'])
                dps=min(dps,incoming*1.2)
                estimated=c['health']-dps*delay
                # A pessimistic prediction must not suppress an available hit:
                # the other attack can still miss, retarget, or land after ours.
                if c['health']<=dealt*.96 or (c['health']<=dealt*1.8 and 0<estimated<=dealt*.95):
                    candidates.append((c['team']==team,abs(estimated-dealt*.65),c))
            if candidates:
                c=min(candidates,key=lambda v:v[:2])[2]
                a['action']=int(Act.ATTACK);self.lock_until=now+me.get('attack_point',.5)+.25
                self.attack_ready=now+me.get('attack_interval',1.5)
                return a,target(c)
        # Raze is only selected if its CURRENT facing/position already yields a hit.
        facing=math.radians(me.get('facing',0))
        for i,range_ in reversed(list(enumerate((200,450,700)))):
            if not skills[i].get('castable') or skills[i].get('level',0)==0: continue
            center={'x':me['x']+math.cos(facing)*range_,'y':me['y']+math.sin(facing)*range_}
            hero_hit=enemy is not None and distance(center,enemy)<215
            creep_kills=sum(distance(center,c)<215 and c['health']<(85+70*(skills[i]['level']-1))*.7 for c in foe_creeps)
            urgent_creep=any(c['health']<damage*2.2 and distance(me,c)<reach+50 for c in foe_creeps)
            if (hero_hit and not urgent_creep and hp>.65 and fraction(enemy,'health')<.35) or (creep_kills>=2 and mp>.55):
                a['action']=int(Act.RAZE_NEAR)+i;self.lock_until=now+.65;return a,me
        urgent_creep=any(c['health']<damage*2.2 and distance(me,c)<reach+50 for c in foe_creeps)
        if enemy and not urgent_creep and foe_distance<reach and hp>.8 and fraction(enemy,'health')<.45 and len(own_creeps)>=len(foe_creeps) and now>=self.attack_ready:
            enemy_t1=s['towers'][2 if team==2 else 0]
            if not enemy_t1.get('visible') or distance(me,enemy_t1)>850:
                a['action']=int(Act.ATTACK);self.lock_until=now+me.get('attack_point',.5)+.1
                self.attack_ready=now+me.get('attack_interval',1.5);return a,target(enemy)
        # The executor requests one attack, so no extra STOP is needed to end it.
        if foe_creeps:
            nearest=min(foe_creeps,key=lambda c:distance(me,c))
            enemy_t1=s['towers'][2 if team==2 else 0]
            tanking=[c for c in own_creeps if distance(c,enemy_t1)<650]
            if distance(me,enemy_t1)<820 and len(tanking)<2:
                dx,dy=me['x']-enemy_t1['x'],me['y']-enemy_t1['y'];length=max(1,math.hypot(dx,dy))
                return move({'x':enemy_t1['x']+dx/length*950,'y':enemy_t1['y']+dy/length*950})
            if distance(me,nearest)>reach-55:
                dx,dy=me['x']-nearest['x'],me['y']-nearest['y'];length=max(1,math.hypot(dx,dy))
                return move({'x':nearest['x']+dx/length*(reach-75),'y':nearest['y']+dy/length*(reach-75)})
            return a,me
        enemy_t1=s['towers'][2 if team==2 else 0]
        allied_front=[c for c in own_creeps if distance(c,enemy_t1)<850] if enemy_t1.get('x') is not None else []
        if allied_front and enemy_t1.get('attackable') and distance(me,enemy_t1)<reach+100 and hp>.55 and now>=self.attack_ready:
            a['action']=int(Act.ATTACK);self.lock_until=now+me.get('attack_point',.5)+.1
            self.attack_ready=now+me.get('attack_interval',1.5);return a,target(enemy_t1)
        # Deterministic opening and regrouping. These decisions become labels;
        # the neural controller has no such fallback.
        goal={'x':-1050 if team==2 else 100,'y':-950 if team==2 else 200}
        if distance(me,goal)>100: return move(goal)
        return a,me
