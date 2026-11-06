"""Exact decision-clock masks for the user-authorized scripted base phase."""
import math
import numpy as np
class MidPhase:
    def __init__(self,manifest):
        self.native=bool(manifest.get('scripted_base_to_mid',False));self.first={};self.start={}
        if self.native:return
        for entry in manifest['files']:
            block=int(entry['file'].split('-')[1])
            for actor in entry['actors']:self.first[actor]=min(self.first.get(actor,block),block)
        for game in manifest['game_metrics']:
            for side,row in enumerate(game['sides']):
                elapsed=row['mid_arrival_elapsed_seconds'];self.start[2*game['game']+side]=None if elapsed is None else math.ceil(elapsed*5-1e-3)
        if set(self.first)-set(self.start):raise ValueError('Missing native mid-arrival metadata')
    def apply(self,entry,actors,b,signals):
        n,t=signals.shape[:2]
        if self.native:return signals[:,:,10].copy(),signals[:,:,11]>.5,np.zeros((n,t),bool)
        block=int(entry['file'].split('-')[1]);allowed=np.zeros((n,t),bool);handoff=np.zeros((n,t),bool)
        for i,actor in enumerate(actors):
            begin=self.start[actor]
            if begin is not None:
                step=(block-self.first[actor])*128+np.arange(t);allowed[i]=step>=begin;handoff[i]=step==begin
        b['loss_valid']&=allowed.reshape(-1)
        resets=signals[:,:,10].copy();resets[handoff]=1
        previous=b['previous'].reshape(n,t,55);previous[handoff]=0;previous[:,:,1][handoff]=1;previous[:,:,49][handoff]=1
        return resets,allowed,handoff
