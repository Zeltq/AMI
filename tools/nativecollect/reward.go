package main

import (
	wire "sfduel/nativecollect/duelwire"
	pb "sfduel/nativecollect/worldstate"
)

func authoritativeReward(s *pb.CMsgBotWorldState, team int, tick int32) *wire.RewardFrame {
	h := ownHero(s, team)
	name := "npc_dota_goodguys_tower1_mid"
	if team == 3 {
		name = "npc_dota_badguys_tower1_mid"
	}
	var tower *pb.CMsgBotWorldState_Unit
	for _, u := range s.Units {
		if u.GetName() == name && int(u.GetTeamId()) == team {
			tower = u
			break
		}
	}
	if h == nil || tower == nil || h.Health == nil || h.HealthMax == nil || h.LastHits == nil || h.Denies == nil || tower.Health == nil || tower.HealthMax == nil {
		return nil
	}
	return &wire.RewardFrame{Team: int32(team), Tick: tick, Time: s.GetDotaTime(), Values: []float32{float32(h.GetLastHits()), float32(h.GetDenies()), float32(h.GetHealth()), float32(h.GetHealthMax()), float32(tower.GetHealth()), float32(tower.GetHealthMax())}}
}
