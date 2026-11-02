package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"google.golang.org/protobuf/proto"
	"io"
	"math"
	"net"
	"os"
	"path/filepath"
	wire "sfduel/nativecollect/duelwire"
	pb "sfduel/nativecollect/worldstate"
	"strconv"
	"strings"
	"sync"
	"time"
)

type modelKey struct {
	team int32
	tick int32
}

type actorTiming struct {
	Calls       uint64  `json:"calls"`
	ModelWaitMS float64 `json:"model_wait_ms_total"`
	RPCMS       float64 `json:"rpc_ms_total"`
	PublishMS   float64 `json:"publish_ms_total"`
	MaxQueue    int     `json:"max_queued_observations"`
}

func (c *duelCapture) controls(team int) bool {
	return c.policyAddress != "" && (c.policyTeam == 0 || c.policyTeam == team)
}
func (c *duelCapture) takeModel(team, tick int32) (*wire.ModelSupplement, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := c.modelFrames[modelKey{team, tick}]
	if m != nil {
		for key := range c.modelFrames {
			if key.team == team && key.tick <= tick {
				delete(c.modelFrames, key)
			}
		}
	}
	return m, c.terminalTime != nil
}

type duelCapture struct {
	mu             sync.Mutex
	policyAddress  string
	policyTeam     int
	allowLate      bool
	modelFrames    map[modelKey]*wire.ModelSupplement
	terminalTime   *float64
	modelParts     map[string]*fragment
	output         *gzip.Writer
	records        chan *wire.Record
	commands       map[int]map[int32]*wire.Decision
	decisions      map[int]map[uint64]*wire.Decision
	acks           map[int]map[uint64]*wire.Execution
	latest         map[int]*pb.CMsgBotWorldState
	aux            map[int][]*wire.Supplement
	errors         []string
	bots, run      string
	period, offset int
	timings        map[int]*actorTiming
}

func (c *duelCapture) write(record *wire.Record) error {
	select {
	case c.records <- record:
		return nil
	default:
		return fmt.Errorf("binary recorder backlog exceeded; rejecting capture")
	}
}
func (c *duelCapture) writeRecord(record *wire.Record) error {
	b, err := proto.Marshal(record)
	if err != nil {
		return err
	}
	var h [4]byte
	binary.LittleEndian.PutUint32(h[:], uint32(len(b)))
	if _, err = c.output.Write(h[:]); err != nil {
		return err
	}
	_, err = c.output.Write(b)
	return err
}
func (c *duelCapture) publish(team int, tick int32, d *wire.Decision, state *pb.CMsgBotWorldState) error {
	queue := c.commands[team]
	queue[d.ExecuteTick] = d
	var b strings.Builder
	b.WriteString("return {\nunits={")
	for _, u := range state.Units {
		if u.GetName() == "npc_dota_hero_nevermore" || laneCreep(u) || u.GetUnitType() == pb.CMsgBotWorldState_TOWER {
			fmt.Fprintf(&b, "%d,", u.GetHandle())
		}
	}
	b.WriteString("},\n")
	fmt.Fprintf(&b, "allow_late=%t,\n", c.allowLate)
	for key, cmd := range queue {
		if key < tick-30 {
			delete(queue, key)
			continue
		}
		fmt.Fprintf(&b, "[%d]={sequence=%d,observed_tick=%d,execute_tick=%d,action=%d,target=%d,x=%.9g,y=%.9g,item=%q,buy=%q},\n", key, cmd.Sequence, cmd.ObservedTick, cmd.ExecuteTick, cmd.Action, cmd.Target, cmd.X, cmd.Y, cmd.Item, cmd.Buy)
	}
	b.WriteString("}\n")
	path := filepath.Join(c.bots, fmt.Sprintf("native_commands_%d.lua", team))
	temp := path + ".tmp"
	if err := os.WriteFile(temp, []byte(b.String()), 0644); err != nil {
		return err
	}
	var err error
	for i := 0; i < 20; i++ {
		err = os.Rename(temp, path)
		if err == nil {
			return nil
		}
		time.Sleep(time.Millisecond)
	}
	return err
}
func (c *duelCapture) actor(ctx context.Context, team, port int) error {
	timing := &actorTiming{}
	c.mu.Lock()
	if c.timings == nil {
		c.timings = map[int]*actorTiming{}
	}
	c.timings[team] = timing
	c.mu.Unlock()
	var conn net.Conn
	var err error
	for ctx.Err() == nil {
		conn, err = net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
		if err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if conn == nil {
		return ctx.Err()
	}
	defer conn.Close()
	// A native snapshot can be tens of KiB. The default Windows receive window
	// holds very few frames at accelerated game speed; absorb short scheduling
	// stalls without making the engine's socket writer disconnect the stream.
	if tcp, ok := conn.(*net.TCPConn); ok {
		if err = tcp.SetReadBuffer(16 * 1024 * 1024); err != nil {
			return err
		}
	}
	go func() { <-ctx.Done(); conn.Close() }()
	reader := bufio.NewReaderSize(conn, 256*1024)
	states := make(chan *pb.CMsgBotWorldState, 1024)
	readErrors := make(chan error, 1)
	readerDone := make(chan struct{})
	defer func() { conn.Close(); <-readerDone }()
	go func() {
		defer close(states)
		defer close(readerDone)
		for {
			var header [4]byte
			if _, e := io.ReadFull(reader, header[:]); e != nil {
				readErrors <- e
				return
			}
			n := binary.LittleEndian.Uint32(header[:])
			if n == 0 || n > 32*1024*1024 {
				readErrors <- fmt.Errorf("bad native frame length")
				return
			}
			data := make([]byte, n)
			if _, e := io.ReadFull(reader, data); e != nil {
				readErrors <- e
				return
			}
			state := new(pb.CMsgBotWorldState)
			if e := proto.Unmarshal(data, state); e != nil {
				readErrors <- e
				return
			}
			if state.GetGameState() != 4 && state.GetGameState() != 5 {
				continue
			}
			if _, e := os.Stat(filepath.Join(c.run, "ready")); e != nil {
				continue
			}
			tick := int32(math.Round(float64(state.GetGameTime()) * 30))
			if tick%int32(c.period) != 0 {
				continue
			}
			c.mu.Lock()
			c.latest[team] = state
			c.mu.Unlock()
			if reward := authoritativeReward(state, team, tick); reward != nil {
				if e := c.write(&wire.Record{RewardFrame: reward}); e != nil {
					readErrors <- e
					return
				}
			}
			select {
			case states <- state:
			case <-ctx.Done():
				return
			default:
				readErrors <- fmt.Errorf("native observation backlog exceeded")
				return
			}
		}
	}()
	teacher := nativeTeacher{team: team}
	var inference net.Conn
	defer func() {
		if inference != nil {
			inference.Close()
		}
	}()
	lastTick := int32(-10000000)
	for {
		var s *pb.CMsgBotWorldState
		select {
		case <-ctx.Done():
			return ctx.Err()
		case readError := <-readErrors:
			return readError
		case value, ok := <-states:
			if !ok {
				select {
				case e := <-readErrors:
					return e
				default:
					return ctx.Err()
				}
			}
			s = value
		}

		if s.GetGameState() != 4 && s.GetGameState() != 5 {
			continue
		}
		if _, err = os.Stat(filepath.Join(c.run, "ready")); err != nil {
			continue
		}
		tick := int32(math.Round(float64(s.GetGameTime()) * 30))

		if tick%int32(c.period) != 0 || tick <= lastTick {
			continue
		}
		h := ownHero(s, team)
		if lastTick != -10000000 && tick != lastTick+int32(c.period) {
			return fmt.Errorf("native decision gap team %d: %d -> %d", team, lastTick, tick)
		}
		lastTick = tick
		c.mu.Lock()
		terminal := c.terminalTime != nil && float64(s.GetDotaTime()) >= *c.terminalTime
		teacher.supp = nil
		for _, aux := range c.aux[team] {
			if aux.Tick <= tick && (teacher.supp == nil || aux.Tick > teacher.supp.Tick) {
				teacher.supp = aux
			}
		}
		c.mu.Unlock()
		var chosen *wire.Decision
		if !terminal && h != nil && h.GetIsAlive() && s.GetDotaTime() < 600 {
			if c.controls(team) {
				if len(states) > timing.MaxQueue {
					timing.MaxQueue = len(states)
				}
				teacher.sequence++
				stub := &wire.Decision{Sequence: teacher.sequence, Team: int32(team), ObservedTick: tick, ExecuteTick: tick + int32(c.offset), ObservedDotaTime: s.GetDotaTime()}
				var model *wire.ModelSupplement
				modelWaitStarted := time.Now()
				deadline := time.Now().Add(time.Second)
				for model == nil && time.Now().Before(deadline) && ctx.Err() == nil {
					var ended bool
					model, ended = c.takeModel(int32(team), tick)
					if ended {
						terminal = true
						break
					}
					if model == nil {
						time.Sleep(time.Millisecond)
					}
				}
				if terminal {
					continue
				}
				if model == nil {
					return fmt.Errorf("policy model supplement missing at %d", tick)
				}
				timing.ModelWaitMS += float64(time.Since(modelWaitStarted)) / float64(time.Millisecond)
				c.mu.Lock()
				teacher.supp = nil
				for _, aux := range c.aux[team] {
					if aux.Tick <= tick && (teacher.supp == nil || aux.Tick > teacher.supp.Tick) {
						teacher.supp = aux
					}
				}
				c.mu.Unlock()
				if inference == nil {
					inference, err = net.DialTimeout("tcp", c.policyAddress, 5*time.Second)
					if err != nil {
						return err
					}
				}
				request := &wire.Record{Observation: s, Decision: stub, Supplement: teacher.supp, ModelSupplement: model, Episode: c.run}
				rpcStarted := time.Now()
				chosen, err = remotePolicy(inference, request)
				timing.RPCMS += float64(time.Since(rpcStarted)) / float64(time.Millisecond)
				timing.Calls++
				if err != nil {
					return err
				}
				if chosen.Sequence != stub.Sequence || chosen.Team != stub.Team || chosen.ObservedTick != stub.ObservedTick || chosen.ExecuteTick != stub.ExecuteTick {
					return fmt.Errorf("policy reply identity mismatch")
				}
			} else {
				chosen = teacher.decide(s, c.offset)
			}
			c.mu.Lock()
			c.decisions[team][chosen.Sequence] = chosen
			err = c.write(&wire.Record{Observation: s, Decision: chosen, Supplement: teacher.supp, Episode: c.run})
			c.mu.Unlock()
		}
		if err == nil && chosen != nil {
			publishStarted := time.Now()
			err = c.publish(team, tick, chosen, s)
			timing.PublishMS += float64(time.Since(publishStarted)) / float64(time.Millisecond)
		}
		if err != nil {
			return err
		}
	}
}
func (c *duelCapture) readAck(line string) error {
	if i := strings.Index(line, "SF_MODEL|"); i >= 0 {
		fields := strings.SplitN(strings.TrimSpace(line[i:]), "|", 6)
		if len(fields) != 6 {
			return fmt.Errorf("truncated model supplement")
		}
		part, e := strconv.Atoi(fields[3])
		if e != nil {
			return e
		}
		count, e := strconv.Atoi(fields[4])
		if e != nil {
			return e
		}
		if part < 1 || part > count || count > 128 {
			return fmt.Errorf("bad model fragments")
		}
		if c.modelParts == nil {
			c.modelParts = map[string]*fragment{}
		}
		key := fields[1] + ":" + fields[2]
		f := c.modelParts[key]
		if f == nil {
			f = &fragment{Count: count, Parts: map[int][]byte{}}
			c.modelParts[key] = f
		}
		if f.Count != count {
			return fmt.Errorf("fragment count changed")
		}
		f.Parts[part] = []byte(fields[5])
		if len(f.Parts) == count {
			var body []byte
			for n := 1; n <= count; n++ {
				body = append(body, f.Parts[n]...)
			}
			m := new(wire.ModelSupplement)
			if e = json.Unmarshal(body, m); e != nil {
				return e
			}
			if fmt.Sprint(m.Team) != fields[1] || fmt.Sprint(m.Tick) != fields[2] {
				return fmt.Errorf("model supplement identity mismatch")
			}
			delete(c.modelParts, key)
			if c.controls(int(m.Team)) {
				c.mu.Lock()
				if c.modelFrames == nil {
					c.modelFrames = map[modelKey]*wire.ModelSupplement{}
				}
				c.modelFrames[modelKey{m.Team, m.Tick}] = m
				overflow := len(c.modelFrames) > 16384
				c.mu.Unlock()
				if overflow {
					return fmt.Errorf("unconsumed model observation backlog exceeded")
				}
			}
			return c.write(&wire.Record{ModelSupplement: m})
		}
		return nil
	}

	if i := strings.Index(line, "SF_NATIVE_AUX|"); i >= 0 {
		aux := new(wire.Supplement)
		if err := json.Unmarshal([]byte(strings.TrimSpace(line[i+14:])), aux); err != nil {
			return err
		}
		if aux.Team != 2 && aux.Team != 3 {
			return fmt.Errorf("bad auxiliary team")
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		if c.aux == nil {
			c.aux = map[int][]*wire.Supplement{}
		}
		values := append(c.aux[int(aux.Team)], aux)
		if len(values) > 128 {
			values = values[len(values)-128:]
		}
		c.aux[int(aux.Team)] = values
		return nil
	}

	start := strings.Index(line, "SF_NATIVE_ACK|")
	if start < 0 {
		return nil
	}
	f := strings.Split(strings.TrimSpace(line[start:]), "|")
	if len(f) != 7 {
		return fmt.Errorf("bad execution acknowledgement")
	}
	team, e1 := strconv.Atoi(f[1])
	seq, e2 := strconv.ParseUint(f[2], 10, 64)
	tick, e3 := strconv.Atoi(f[3])
	target, e4 := strconv.ParseUint(f[5], 10, 32)
	if e1 != nil || e2 != nil || e3 != nil || e4 != nil || (team != 2 && team != 3) {
		return fmt.Errorf("bad acknowledgement identity")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	d := c.decisions[team][seq]
	if d == nil {
		return fmt.Errorf("execution has no causal decision")
	}
	if !validExecutionTick(int(d.ExecuteTick), tick, c.allowLate) {
		return fmt.Errorf("execution tick differs from schedule")
	}
	if c.acks[team][seq] != nil {
		return fmt.Errorf("duplicate execution")
	}
	a := &wire.Execution{Sequence: seq, Team: int32(team), Tick: int32(tick), Accepted: f[4] == "1", ActualTarget: uint32(target), Reason: f[6]}
	c.acks[team][seq] = a
	return c.write(&wire.Record{Execution: a})
}
func validateTiming(period, offset int) error {
	if period != 4 && period != 2 {
		return fmt.Errorf("period must be 2 or 4 engine ticks")
	}
	if offset < period {
		return fmt.Errorf("offset must be at least the observation period")
	}
	return nil
}

func validExecutionTick(planned, actual int, allowLate bool) bool {
	return actual == planned || (allowLate && actual > planned)
}

func duel(run, bots, log string, ports [2]int, period, offset int, timeout, until float64, policyAddress string, policyTeam int, allowLate bool) error {
	if err := validateTiming(period, offset); err != nil {
		return err
	}
	os.MkdirAll(run, 0755)
	file, err := os.Create(filepath.Join(run, "native-decisions.pb.gz"))
	if err != nil {
		return err
	}
	defer file.Close()
	z, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
	c := &duelCapture{output: z, bots: bots, run: run, period: period, offset: offset, commands: map[int]map[int32]*wire.Decision{}, decisions: map[int]map[uint64]*wire.Decision{}, acks: map[int]map[uint64]*wire.Execution{}, latest: map[int]*pb.CMsgBotWorldState{}}
	c.allowLate = allowLate
	c.policyAddress = policyAddress
	c.policyTeam = policyTeam
	c.records = make(chan *wire.Record, 256)
	writerDone := make(chan error, 1)
	go func() {
		var failure error
		for record := range c.records {
			if failure == nil {
				failure = c.writeRecord(record)
			}
		}
		if e := z.Close(); failure == nil {
			failure = e
		}
		writerDone <- failure
	}()
	for _, team := range []int{2, 3} {
		c.commands[team] = map[int32]*wire.Decision{}
		c.decisions[team] = map[uint64]*wire.Decision{}
		c.acks[team] = map[uint64]*wire.Execution{}
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout*float64(time.Second)))
	defer cancel()
	var wg sync.WaitGroup
	fail := make(chan error, 2)
	for i, team := range []int{2, 3} {
		wg.Add(1)
		go func(team, port int) {
			defer wg.Done()
			if e := c.actor(ctx, team, port); e != nil && ctx.Err() == nil {
				fail <- e
			}
		}(team, ports[i])
	}
	started := time.Now()
	position := int64(0)
	buffer := ""
	reason := ""
	referee := &recorder{Last: map[int]*frame{}}
	var finalOutcome *outcome
	terminalTime := float64(0)
	for reason == "" && err == nil {
		select {
		case err = <-fail:
		case <-ctx.Done():
			err = ctx.Err()
		default:
		}
		if f, e := os.Open(log); e == nil {
			f.Seek(position, 0)
			b, _ := io.ReadAll(f)
			position += int64(len(b))
			f.Close()
			buffer += string(b)
			lines := strings.Split(buffer, "\n")
			buffer = lines[len(lines)-1]
			for _, line := range lines[:len(lines)-1] {
				if e = c.readAck(line); e != nil {
					err = e
					break
				}
				if strings.Contains(line, "SF_NATIVE_ERROR|") {
					err = fmt.Errorf("%s", line)
					break
				}
				if i := strings.Index(line, "SF_NATIVE_END|"); i >= 0 {
					parts := strings.Split(strings.TrimSpace(line[i:]), "|")
					if len(parts) == 5 {
						when, _ := strconv.ParseFloat(parts[2], 64)
						team, _ := strconv.Atoi(parts[1])
						lh, _ := strconv.Atoi(parts[4])
						referee.Events = append(referee.Events, terminal{Team: team, Time: when, Reason: parts[3], LH: lh})
						c.mu.Lock()
						if c.terminalTime == nil || when < *c.terminalTime {
							value := when
							c.terminalTime = &value
						}
						c.mu.Unlock()
					}
				}
			}
		}
		c.mu.Lock()
		a, b := c.latest[2], c.latest[3]
		for _, team := range []int{2, 3} {
			state := c.latest[team]
			if state == nil {
				continue
			}
			referee.Last[team] = &frame{Time: float64(state.GetDotaTime())}
			if state.GetDotaTime() >= 0 {
				for _, player := range state.Players {
					if int(player.GetTeamId()) == team && player.GetHeroId() == 11 && player.GetDeaths() > 0 {
						already := false
						for _, event := range referee.Events {
							if event.Team == team && event.Reason == "death" {
								already = true
							}
						}
						if !already {
							referee.Events = append(referee.Events, terminal{Team: team, Time: float64(state.GetDotaTime()), Reason: "death"})
						}
					}
				}
			}
		}
		if candidate := referee.result(); candidate != nil {
			if c.terminalTime == nil {
				value := candidate.Time
				c.terminalTime = &value
			}
			if referee.ready(candidate) {
				finalOutcome = candidate
				reason = candidate.Reason
				terminalTime = candidate.Time
			}
		}
		if a != nil && b != nil && until < 600 && math.Min(float64(a.GetDotaTime()), float64(b.GetDotaTime())) >= until {
			reason = "diagnostic_cut"
			terminalTime = until
		}
		c.mu.Unlock()
		time.Sleep(2 * time.Millisecond)
	}
	cancel()
	wg.Wait()
	close(c.records)
	if e := <-writerDone; err == nil {
		err = e
	}
	c.mu.Lock()
	stats := map[int]map[string]any{}
	missing := 0
	for _, team := range []int{2, 3} {
		rejected := 0
		late := 0
		maxLate := int32(0)
		teamMissing := 0
		missedTicks := []int32{}
		for seq, d := range c.decisions[team] {
			if terminalTime != 0 && float64(d.ObservedDotaTime)+float64(c.offset)/30 >= terminalTime {
				continue
			}
			a := c.acks[team][seq]
			if a == nil {
				teamMissing++
				missedTicks = append(missedTicks, d.ExecuteTick)
			} else if !a.Accepted {
				rejected++
			}
			if a != nil && a.Tick > d.ExecuteTick {
				late++
				missedTicks = append(missedTicks, d.ExecuteTick)
				if a.Tick-d.ExecuteTick > maxLate {
					maxLate = a.Tick - d.ExecuteTick
				}
			}
		}
		h := ownHero(c.latest[team], team)
		stats[team] = map[string]any{"decisions": len(c.decisions[team]), "acknowledged": len(c.acks[team]), "missed_deadlines": teamMissing + late, "rejected": rejected}
		stats[team]["missed_ticks"] = missedTicks
		stats[team]["late_executions"] = late
		stats[team]["max_lateness_ticks"] = maxLate
		stats[team]["unacknowledged"] = teamMissing
		if h != nil {
			stats[team]["last_hits"] = h.GetLastHits()
			stats[team]["denies"] = h.GetDenies()
			stats[team]["level"] = h.GetLevel()
		}
		missing += teamMissing
	}

	report := map[string]any{"reason": reason, "time": terminalTime, "outcome": finalOutcome, "clock": "game_time", "period_ticks": period, "offset_ticks": offset, "wall_seconds": time.Since(started).Seconds(), "sides": stats, "training_started": false, "experimental": true, "eligible": false, "unacknowledged_commands": missing, "missing_ack_policy": "keep observations and recurrent sequence; mask action supervision on unacknowledged steps; never execute stale commands"}
	report["allow_late"] = allowLate
	report["performance"] = c.timings
	if allowLate {
		report["missing_ack_policy"] = "execute newest due command even when late; never execute an older command after a newer sequence; record actual execution tick"
	}
	if err != nil {
		report["error"] = err.Error()
	}
	b, _ := json.MarshalIndent(report, "", "  ")
	os.WriteFile(filepath.Join(run, "duel-result.json"), b, 0644)
	fmt.Println(string(b))
	c.mu.Unlock()
	return err
}

func remotePolicy(conn net.Conn, request *wire.Record) (*wire.Decision, error) {
	conn.SetDeadline(time.Now().Add(5 * time.Second))
	payload, err := proto.Marshal(request)
	if err != nil {
		return nil, err
	}
	frame := make([]byte, 4+len(payload))
	binary.LittleEndian.PutUint32(frame, uint32(len(payload)))
	copy(frame[4:], payload)
	for len(frame) > 0 {
		n, e := conn.Write(frame)
		if e != nil {
			return nil, e
		}
		frame = frame[n:]
	}
	var header [4]byte
	if _, err = io.ReadFull(conn, header[:]); err != nil {
		return nil, err
	}
	size := binary.LittleEndian.Uint32(header[:])
	if size == 0 || size > 1000000 {
		return nil, fmt.Errorf("bad policy response size")
	}
	body := make([]byte, size)
	if _, err = io.ReadFull(conn, body); err != nil {
		return nil, err
	}
	decision := new(wire.Decision)
	if err = proto.Unmarshal(body, decision); err != nil {
		return nil, err
	}
	return decision, nil
}
