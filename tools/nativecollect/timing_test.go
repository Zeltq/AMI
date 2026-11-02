package main

import (
	wire "sfduel/nativecollect/duelwire"
	"testing"
)

func TestModelFramesRetainedUntilConsumedAndSeparatedByTeam(t *testing.T) {
	c := &duelCapture{policyAddress: "test", policyTeam: 0, modelFrames: map[modelKey]*wire.ModelSupplement{}}
	for tick := int32(1744); tick <= 1992; tick += 4 {
		for _, team := range []int32{2, 3} {
			c.modelFrames[modelKey{team, tick}] = &wire.ModelSupplement{Team: team, Tick: tick}
		}
	}
	m, ended := c.takeModel(2, 1744)
	if m == nil || m.Team != 2 || ended {
		t.Fatal("queued frame was lost")
	}
	if c.modelFrames[modelKey{3, 1744}] == nil {
		t.Fatal("other player's frame removed")
	}
	if c.modelFrames[modelKey{2, 1748}] == nil {
		t.Fatal("future frame removed")
	}
	end := 196.133
	c.terminalTime = &end
	if _, ended = c.takeModel(2, 10000); !ended {
		t.Fatal("waiting for nonexistent post-terminal observation")
	}
	if !c.controls(2) || !c.controls(3) {
		t.Fatal("self-play must control both teams")
	}
}

func TestLateExecutionAcknowledgementRecordsActualTick(t *testing.T) {
	c := &duelCapture{allowLate: true, records: make(chan *wire.Record, 2),
		decisions: map[int]map[uint64]*wire.Decision{2: {1: {Sequence: 1, Team: 2, ObservedTick: 100, ExecuteTick: 106}}},
		acks:      map[int]map[uint64]*wire.Execution{2: {}}}
	if err := c.readAck("SF_NATIVE_ACK|2|1|108|1|0|"); err != nil {
		t.Fatal(err)
	}
	if c.acks[2][1].Tick != 108 {
		t.Fatal("actual late execution tick lost")
	}
	if err := c.readAck("SF_NATIVE_ACK|2|1|109|1|0|"); err == nil {
		t.Fatal("duplicate execution accepted")
	}
}

func TestExecutionAcknowledgementClock(t *testing.T) {
	for _, actual := range []int{106, 107, 108, 150} {
		if !validExecutionTick(106, actual, true) {
			t.Fatal("late acknowledgement rejected", actual)
		}
	}
	if validExecutionTick(106, 105, true) || validExecutionTick(106, 107, false) {
		t.Fatal("invalid execution accepted")
	}
}

func TestExecutionOffsetNeedNotMatchObservationGrid(t *testing.T) {
	for _, offset := range []int{4, 6, 7, 8, 10} {
		if err := validateTiming(4, offset); err != nil {
			t.Fatalf("offset %d rejected: %v", offset, err)
		}
	}
	for _, clock := range [][2]int{{4, 3}, {3, 6}, {0, 7}} {
		if validateTiming(clock[0], clock[1]) == nil {
			t.Fatalf("invalid clock accepted: %v", clock)
		}
	}
}
