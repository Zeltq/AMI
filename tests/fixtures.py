import copy
from sf_duel.spec import *

def snapshot():
    own={'id':1,'name':'npc_dota_hero_nevermore','type':'hero','visible':True,'team':2,
         'x':0,'y':0,'z':128,'health':600,'health_max':600,'mana':300,'mana_max':300,
         'gold':100,'last_hits':0,'attack_damage':50,'attack_point':.5,'attack_interval':1.6}
    enemy=dict(own,id=2,team=3,x=600,gold=99999,health=500,attackable=True)
    towers=[{'id':i+10,'name':name,'type':'tower','team':2 if i<2 else 3,
             'visible':True,'health':1800,'health_max':1800,'x':i*500,'y':i*500,
             'tp':i<2,'attackable':i>=2} for i,name in enumerate(TOWERS)]
    return dict(version=1,match='fixture',seq=1,team=2,player=0,time=0,side=1,
        self=own,enemy=enemy,towers=towers,creeps=[],trees=[],projectiles=[],
        skills=[dict(level=1,cooldown=0,mana_cost=75,castable=True) for _ in SPELLS],
        items=[dict(inventory=0,castable=False) for _ in ITEMS],
        enemy_items=[dict(inventory=0,backpack=0) for _ in ITEMS],
        terrain=[[[0. for _ in range(8)] for _ in range(8)] for _ in range(4)],
        base={'id':-1,'type':'base','tp':True,'x':-6500,'y':-6500},next_permanent=0,
        legal={'actions':[True]*10+[False],'items':[False]*8,'buys':[True]+[False]*6})
