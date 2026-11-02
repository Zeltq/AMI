package main

import (
	"math"
	wire "sfduel/nativecollect/duelwire"
	pb "sfduel/nativecollect/worldstate"
	"strings"
)

type nativeTeacher struct {
	team        int
	sequence    uint64
	commitUntil float64
	supp        *wire.Supplement
	nextShop    float64
}

func distance(a, b *pb.CMsgBotWorldState_Unit) float64 {
	return math.Hypot(float64(a.GetLocation().GetX()-b.GetLocation().GetX()), float64(a.GetLocation().GetY()-b.GetLocation().GetY()))
}
func laneCreep(u *pb.CMsgBotWorldState_Unit) bool {
	n := u.GetName()
	return strings.HasPrefix(n, "npc_dota_creep_") || n == "npc_dota_goodguys_siege" || n == "npc_dota_badguys_siege"
}
func physical(damage, armor float64) float64 { return damage * (1 - .06*armor/(1+.06*math.Abs(armor))) }
func modifier(u *pb.CMsgBotWorldState_Unit, name string) bool {
	for _, m := range u.GetModifiers() {
		if m.GetName() == name {
			return true
		}
	}
	return false
}
func ownHero(s *pb.CMsgBotWorldState, team int) *pb.CMsgBotWorldState_Unit {
	for _, u := range s.GetUnits() {
		if u.GetName() == "npc_dota_hero_nevermore" && int(u.GetTeamId()) == team {
			return u
		}
	}
	return nil
}
func inventory(u *pb.CMsgBotWorldState_Unit, id int32) *pb.CMsgBotWorldState_Ability {
	for _, a := range u.GetItems() {
		if a.GetAbilityId() == id {
			return a
		}
	}
	return nil
}
func ownedItem(s *pb.CMsgBotWorldState, h *pb.CMsgBotWorldState_Unit, id int32) bool {
	if inventory(h, id) != nil {
		return true
	}
	for _, u := range s.Units {
		if u.GetTeamId() == h.GetTeamId() && u.GetUnitType() == pb.CMsgBotWorldState_COURIER && inventory(u, id) != nil {
			return true
		}
	}
	return false
}
func ability(u *pb.CMsgBotWorldState_Unit, id int32) *pb.CMsgBotWorldState_Ability {
	for _, a := range u.GetAbilities() {
		if a.GetAbilityId() == id {
			return a
		}
	}
	return nil
}
func available(a *pb.CMsgBotWorldState_Ability) bool {
	return a != nil && a.GetLevel() > 0 && a.GetCooldownRemaining() <= 0 && a.GetIsFullyCastable()
}
func moveCommand(d *wire.Decision, h *pb.CMsgBotWorldState_Unit, x, y float64) {
	dx, dy := x-float64(h.GetLocation().GetX()), y-float64(h.GetLocation().GetY())
	scale := math.Max(1, math.Max(math.Abs(dx), math.Abs(dy))/1024)
	d.Action = 1
	d.X = float32(math.Round(dx/scale/16)*16 + float64(h.GetLocation().GetX()))
	d.Y = float32(math.Round(dy/scale/16)*16 + float64(h.GetLocation().GetY()))
}
func (t *nativeTeacher) decide(s *pb.CMsgBotWorldState, offset int) *wire.Decision {
	t.sequence++
	tick := int32(math.Round(float64(s.GetGameTime()) * 30))
	d := &wire.Decision{Sequence: t.sequence, Team: int32(t.team), ObservedTick: tick, ExecuteTick: tick + int32(offset), ObservedDotaTime: s.GetDotaTime()}
	h := ownHero(s, t.team)
	if h == nil || !h.GetIsAlive() {
		return d
	}
	now := float64(s.GetDotaTime())
	executeAt := now + float64(offset)/30

	hp := float64(h.GetHealth()) / math.Max(1, float64(h.GetHealthMax()))
	missingHP := h.GetHealthMax() - h.GetHealth()
	missingMana := h.GetManaMax() - h.GetMana()
	if now >= t.nextShop {
		t.nextShop = now + .5
		gold := h.GetReliableGold() + h.GetUnreliableGold()
		canBuy := func(name string) bool {
			if t.supp == nil {
				return false
			}
			for _, i := range t.supp.Shop {
				if i.Name == name {
					unlimited := name == "item_boots" || name == "item_bracer" || name == "item_tpscroll"
					return (unlimited || i.Stock != 0) && gold >= i.Price
				}
			}
			return false
		}
		// Stock, price and final purchase legality are rechecked at the engine boundary.
		if now < 0 && !ownedItem(s, h, 44) && canBuy("item_tango") {
			d.Buy = "item_tango"
		} else if missingHP > 180 && !ownedItem(s, h, 39) && canBuy("item_flask") && !modifier(h, "modifier_flask_healing") {
			d.Buy = "item_flask"
		} else if missingMana > 120 && !ownedItem(s, h, 216) && canBuy("item_enchanted_mango") {
			d.Buy = "item_enchanted_mango"
		} else if missingHP > 100 && !ownedItem(s, h, 44) && canBuy("item_tango") && !modifier(h, "modifier_tango_heal") {
			d.Buy = "item_tango"
		} else if !ownedItem(s, h, 29) && canBuy("item_boots") {
			d.Buy = "item_boots"
		} else if ownedItem(s, h, 29) && !ownedItem(s, h, 73) && canBuy("item_bracer") {
			d.Buy = "item_bracer"
		}
	}
	if modifier(h, "modifier_nevermore_requiem_fear") || h.GetIsChanneling() || h.GetIsUsingAbility() || executeAt < t.commitUntil {
		return d
	}
	enemies := []*pb.CMsgBotWorldState_Unit{}
	dangerTower := false
	for _, u := range s.GetUnits() {
		if !u.GetIsAlive() || u.GetTeamId() == h.GetTeamId() {
			continue
		}
		if u.GetName() == "npc_dota_hero_nevermore" {
			enemies = append(enemies, u)
		}
		if strings.Contains(u.GetName(), "tower") && distance(h, u) < 850 {
			dangerTower = true
		}
	}
	nearEnemy := false
	for _, e := range enemies {
		if distance(h, e) < 900 {
			nearEnemy = true
		}
	}
	use := func(id int32, name string) bool {
		a := inventory(h, id)
		if a != nil && (a.GetSlot() <= 5 || (id == 46 && a.GetSlot() == 15)) && a.GetCooldownRemaining() == 0 && a.GetIsFullyCastable() {
			d.Action = 10
			d.Item = name
			return true
		}
		return false
	}
	if missingMana >= 100 && use(216, "item_enchanted_mango") {
		return d
	}
	if missingHP > 200 && !nearEnemy && !modifier(h, "modifier_flask_healing") && use(39, "item_flask") {
		return d
	}
	if missingMana > 150 && !nearEnemy && !modifier(h, "modifier_clarity_potion") && use(38, "item_clarity") {
		return d
	}
	if missingHP > 120 && !modifier(h, "modifier_tango_heal") && t.supp != nil {
		var tree *wire.VisibleTree
		best := 500.0
		for _, v := range t.supp.Trees {
			dist := math.Hypot(float64(v.X-h.GetLocation().GetX()), float64(v.Y-h.GetLocation().GetY()))
			if dist < best {
				best = dist
				tree = v
			}
		}
		if tree != nil && use(44, "item_tango") {
			d.Target = tree.Id
			t.commitUntil = executeAt + best/math.Max(1, float64(h.GetCurrentMovementSpeed())) + .2
			return d
		}
	}
	if hp < .25 {
		for _, u := range s.Units {
			if u.GetName() == "dota_fountain" && u.GetTeamId() == h.GetTeamId() && distance(h, u) > 2500 && use(46, "item_tpscroll") {
				d.X = u.GetLocation().GetX()
				d.Y = u.GetLocation().GetY()
				t.commitUntil = executeAt + 3.2
				return d
			}
		}
		if t.team == 2 {
			moveCommand(d, h, -7542, -6171)
		} else {
			moveCommand(d, h, 7200, 6500)
		}
		return d
	}
	// Raze centers are forward-facing; never auto-aim or silently replace a target.
	face := float64(h.GetFacing()) * math.Pi / 180
	for _, e := range enemies {
		if distance(h, e) > 1000 {
			continue
		}
		for i, id := range []int32{5059, 5060, 5061} {
			a := ability(h, id)
			if !available(a) {
				continue
			}
			r := float64(200 + i*250)
			cx := float64(h.GetLocation().GetX()) + math.Cos(face)*r
			cy := float64(h.GetLocation().GetY()) + math.Sin(face)*r
			if math.Hypot(cx-float64(e.GetLocation().GetX()), cy-float64(e.GetLocation().GetY())) < 200 {
				d.Action = int32(6 + i)
				t.commitUntil = executeAt + .6
				return d
			}
		}
	}
	if available(ability(h, 5064)) {
		for _, e := range enemies {
			if distance(h, e) < 300 && hp > .35 {
				d.Action = 9
				t.commitUntil = executeAt + 1.8
				return d
			}
		}
	}
	var best *pb.CMsgBotWorldState_Unit
	bestScore := -math.MaxFloat64
	for _, u := range s.GetUnits() {
		if !laneCreep(u) || !u.GetIsAlive() || u.GetIsAttackImmune() || distance(h, u) > float64(h.GetAttackRange())+200 {
			continue
		}
		ally := u.GetTeamId() == h.GetTeamId()
		if ally && float64(u.GetHealth()) > float64(u.GetHealthMax())*.5 {
			continue
		}
		damage := physical(float64(h.GetAttackDamage()), float64(u.GetArmor()))
		hp := float64(u.GetHealth())
		travel := distance(h, u)/math.Max(1, float64(h.GetAttackProjectileSpeed())) + float64(h.GetAttackAnimPoint())/math.Max(.1, float64(h.GetAttackSpeed())) + float64(offset)/30
		predicted := hp
		for _, shot := range u.GetIncomingTrackingProjectiles() {
			if !shot.GetIsAttack() || shot.GetVelocity() == 0 {
				continue
			}
			arrival := math.Hypot(float64(shot.GetLocation().GetX()-u.GetLocation().GetX()), float64(shot.GetLocation().GetY()-u.GetLocation().GetY())) / float64(shot.GetVelocity())
			if arrival > travel {
				continue
			}
			for _, caster := range s.GetUnits() {
				if caster.GetHandle() == shot.GetCasterHandle() {
					predicted -= physical(float64(caster.GetAttackDamage()), float64(u.GetArmor()))
					break
				}
			}
		}
		if predicted > 0 && predicted <= damage {
			score := 1000 - hp - distance(h, u)*.1
			if !ally {
				score += 300
			}
			if score > bestScore {
				bestScore = score
				best = u
			}
		}
	}
	if best != nil {
		d.Action = 3
		d.Target = best.GetHandle()
		t.commitUntil = executeAt + float64(h.GetAttackAnimPoint())/math.Max(.1, float64(h.GetAttackSpeed())) + .04
		return d
	}
	if !dangerTower && hp > .5 {
		for _, e := range enemies {
			if distance(h, e) <= float64(h.GetAttackRange()) && hp >= float64(e.GetHealth())/math.Max(1, float64(e.GetHealthMax()))-.15 {
				d.Action = 3
				d.Target = e.GetHandle()
				t.commitUntil = executeAt + float64(h.GetAttackAnimPoint())/math.Max(.1, float64(h.GetAttackSpeed())) + .04
				return d
			}
		}
	}
	// Position behind the nearby enemy wave. Ignore side lanes and neutral camps.
	var front *pb.CMsgBotWorldState_Unit
	nearest := 1800.0
	for _, u := range s.GetUnits() {
		if laneCreep(u) && u.GetTeamId() != h.GetTeamId() && u.GetIsAlive() {
			dist := distance(h, u)
			if dist < nearest {
				nearest = dist
				front = u
			}
		}
	}
	x, y := -500.0, -400.0
	if t.team == 3 {
		x, y = 100, 200
	}
	if front != nil {
		x, y = float64(front.GetLocation().GetX()), float64(front.GetLocation().GetY())
		sign := -1.0
		if t.team == 3 {
			sign = 1
		}
		x += sign * 270
		y += sign * 270
	}
	if math.Hypot(x-float64(h.GetLocation().GetX()), y-float64(h.GetLocation().GetY())) > 60 {
		moveCommand(d, h, x, y)
	}
	return d
}
