package main

import (
	"google.golang.org/protobuf/proto"
	wire "sfduel/nativecollect/duelwire"
	pb "sfduel/nativecollect/worldstate"
	"testing"
)

func testHero() *pb.CMsgBotWorldState_Unit {
	return &pb.CMsgBotWorldState_Unit{Handle: proto.Uint32(10), Name: proto.String("npc_dota_hero_nevermore"), TeamId: proto.Uint32(2), IsAlive: proto.Bool(true), Health: proto.Int32(582), HealthMax: proto.Int32(582), Mana: proto.Int32(200), ManaMax: proto.Int32(200), AttackDamage: proto.Int32(50), AttackRange: proto.Int32(525), AttackSpeed: proto.Float32(1), AttackAnimPoint: proto.Float32(.5), AttackProjectileSpeed: proto.Int32(1200), Location: &pb.CMsgBotWorldState_Vector{X: proto.Float32(0), Y: proto.Float32(0)}}
}
func TestNativeTargetAndFixedDelay(t *testing.T) {
	h := testHero()
	creep := &pb.CMsgBotWorldState_Unit{Handle: proto.Uint32(21), Name: proto.String("npc_dota_creep_badguys_melee"), TeamId: proto.Uint32(3), IsAlive: proto.Bool(true), Health: proto.Int32(35), HealthMax: proto.Int32(550), Location: &pb.CMsgBotWorldState_Vector{X: proto.Float32(400)}}
	s := &pb.CMsgBotWorldState{TeamId: proto.Uint32(2), GameTime: proto.Float32(100), DotaTime: proto.Float32(100), Units: []*pb.CMsgBotWorldState_Unit{h, creep}}
	teacher := nativeTeacher{team: 2}
	d := teacher.decide(s, 4)
	if d.Action != 3 || d.Target != 21 || d.ObservedTick != 3000 || d.ExecuteTick != 3004 {
		t.Fatal(d)
	}
	creep.TeamId = proto.Uint32(4)
	creep.Name = proto.String("npc_dota_neutral_centaur_khan")
	teacher = nativeTeacher{team: 2}
	if d = teacher.decide(s, 4); d.Action == 3 {
		t.Fatal("neutral farm selected")
	}
}
func TestNativeRejectsWrongExecutionTick(t *testing.T) {
	d := &wire.Decision{Sequence: 1, Team: 2, ObservedTick: 100, ExecuteTick: 104}
	c := &duelCapture{decisions: map[int]map[uint64]*wire.Decision{2: {1: d}}, acks: map[int]map[uint64]*wire.Execution{2: {}}, records: make(chan *wire.Record, 2)}
	if c.readAck("SF_NATIVE_ACK|2|1|105|1|0|") == nil {
		t.Fatal("late execution accepted")
	}
	if err := c.readAck("SF_NATIVE_ACK|2|1|104|1|0|"); err != nil {
		t.Fatal(err)
	}
	if c.readAck("SF_NATIVE_ACK|2|1|104|1|0|") == nil {
		t.Fatal("duplicate accepted")
	}
}
func TestNativeRecorderBacklogIsExplicit(t *testing.T) {
	c := &duelCapture{records: make(chan *wire.Record, 1)}
	if err := c.write(&wire.Record{}); err != nil {
		t.Fatal(err)
	}
	if c.write(&wire.Record{}) == nil {
		t.Fatal("full recorder queue silently dropped data")
	}
}

func TestNativeShopUsesCurrentPriceAndStock(t *testing.T) {
	h := testHero()
	h.UnreliableGold = proto.Int32(500)
	h.ReliableGold = proto.Int32(0)
	s := &pb.CMsgBotWorldState{TeamId: proto.Uint32(2), GameTime: proto.Float32(100), DotaTime: proto.Float32(100), Units: []*pb.CMsgBotWorldState_Unit{h}}
	teacher := nativeTeacher{team: 2, supp: &wire.Supplement{Shop: []*wire.ShopItem{{Name: "item_boots", Price: 500, Stock: 0}}}}
	if d := teacher.decide(s, 4); d.Buy != "item_boots" {
		t.Fatal("unlimited permanent item treated as sold out", d)
	}
	h.Mana = proto.Int32(0)
	h.UnreliableGold = proto.Int32(100)
	teacher = nativeTeacher{team: 2, supp: &wire.Supplement{Shop: []*wire.ShopItem{{Name: "item_enchanted_mango", Price: 65, Stock: 0}}}}
	if d := teacher.decide(s, 4); d.Buy != "" {
		t.Fatal("sold-out consumable selected", d)
	}
}
func TestNativeSupplementParsing(t *testing.T) {
	c := &duelCapture{}
	if err := c.readAck(`SF_NATIVE_AUX|{"team":2,"tick":100,"shop":[{"name":"item_flask","price":100,"stock":5}],"trees":[{"id":17,"x":50,"y":60,"z":128}]}`); err != nil {
		t.Fatal(err)
	}
	if c.aux[2][0].Shop[0].Price != 100 || c.aux[2][0].Trees[0].Id != 17 {
		t.Fatal("supplement lost fields")
	}
}
