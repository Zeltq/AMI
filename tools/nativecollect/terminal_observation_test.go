package main

import (
	"context"
	"encoding/binary"
	"google.golang.org/protobuf/proto"
	"io"
	"net"
	"os"
	"path/filepath"
	wire "sfduel/nativecollect/duelwire"
	pb "sfduel/nativecollect/worldstate"
	"testing"
	"time"
)

func TestNativeReceiverEOFIsFailureNotTerminal(t *testing.T) {
	world, _ := net.Listen("tcp", "127.0.0.1:0")
	defer world.Close()
	go func() {
		conn, e := world.Accept()
		if e == nil {
			conn.Close()
		}
	}()
	c := &duelCapture{run: t.TempDir(), period: 4}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := c.actor(ctx, 2, world.Addr().(*net.TCPAddr).Port); err != io.EOF {
		t.Fatalf("expected transport EOF, got %v", err)
	}
}

func TestReceiverAdvancesPastDeathWithoutWaitingForPostTerminalModelFrame(t *testing.T) {
	world, _ := net.Listen("tcp", "127.0.0.1:0")
	defer world.Close()
	policy, _ := net.Listen("tcp", "127.0.0.1:0")
	defer policy.Close()
	firstDecision := make(chan struct{})
	go func() {
		conn, e := policy.Accept()
		if e != nil {
			return
		}
		defer conn.Close()
		var header [4]byte
		if _, e = io.ReadFull(conn, header[:]); e != nil {
			return
		}
		body := make([]byte, binary.LittleEndian.Uint32(header[:]))
		if _, e = io.ReadFull(conn, body); e != nil {
			return
		}
		r := new(wire.Record)
		proto.Unmarshal(body, r)
		reply, _ := proto.Marshal(r.Decision)
		binary.LittleEndian.PutUint32(header[:], uint32(len(reply)))
		conn.Write(header[:])
		conn.Write(reply)
		close(firstDecision)
	}()
	run := t.TempDir()
	os.WriteFile(filepath.Join(run, "ready"), []byte("ready"), 0644)
	c := &duelCapture{policyAddress: policy.Addr().String(), policyTeam: 3, bots: run, run: run, period: 4, offset: 6,
		records: make(chan *wire.Record, 32), latest: map[int]*pb.CMsgBotWorldState{},
		modelFrames: map[modelKey]*wire.ModelSupplement{{3, 9996}: {Team: 3, Tick: 9996}},
		decisions:   map[int]map[uint64]*wire.Decision{3: {}}, commands: map[int]map[int32]*wire.Decision{3: {}}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- c.actor(ctx, 3, world.Addr().(*net.TCPAddr).Port) }()
	conn, e := world.Accept()
	if e != nil {
		t.Fatal(e)
	}
	defer conn.Close()
	send := func(tick int32) {
		s := &pb.CMsgBotWorldState{GameState: proto.Uint32(5), GameTime: proto.Float32(float32(tick) / 30), DotaTime: proto.Float32(196 + float32(tick-9996)/30),
			Units: []*pb.CMsgBotWorldState_Unit{{Name: proto.String("npc_dota_hero_nevermore"), TeamId: proto.Uint32(3), IsAlive: proto.Bool(true), Handle: proto.Uint32(1)}}}
		body, _ := proto.Marshal(s)
		var header [4]byte
		binary.LittleEndian.PutUint32(header[:], uint32(len(body)))
		conn.Write(header[:])
		conn.Write(body)
	}
	send(9996)
	select {
	case <-firstDecision:
	case <-time.After(2 * time.Second):
		t.Fatal("initial decision missing")
	}
	// Lua emits END at 10000 and no more model frames. The native stream continues.
	send(10000)
	send(10004)
	deadline := time.Now().Add(500 * time.Millisecond)
	advanced := false
	for time.Now().Before(deadline) {
		c.mu.Lock()
		s := c.latest[3]
		advanced = s != nil && s.GetDotaTime() > 196.14
		c.mu.Unlock()
		if advanced {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !advanced {
		t.Fatal("native receiver stalled behind missing terminal supplement")
	}
	c.mu.Lock()
	end := 196.133
	c.terminalTime = &end
	c.mu.Unlock()
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("actor did not exit")
	}
}
