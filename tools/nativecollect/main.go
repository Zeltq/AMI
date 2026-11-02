// Read native Dota streams without a Python GIL or Lua field-by-field JSON.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"
	pb "sfduel/nativecollect/worldstate"
)

type selfRow struct {
	Time     float64 `json:"time"`
	GameTime float64 `json:"game_time"`
	State    uint32  `json:"state"`
	Team     uint32  `json:"team"`
	Handle   uint32  `json:"handle"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Health   int32   `json:"health"`
	Mana     int32   `json:"mana"`
	Action   uint32  `json:"action_type"`
}

func receive(run string, team, port int) (map[string]any, error) {
	deadline := time.Now().Add(120 * time.Second)
	var conn net.Conn
	var err error
	for time.Now().Before(deadline) {
		conn, err = net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
		if err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	file, err := os.Create(filepath.Join(run, fmt.Sprintf("world-%d.pb.gz", team)))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	compressed, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
	defer compressed.Close()
	rows, err := os.Create(filepath.Join(run, fmt.Sprintf("self-%d.jsonl.gz", team)))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	rowzip, _ := gzip.NewWriterLevel(rows, gzip.BestSpeed)
	defer rowzip.Close()
	encoder := json.NewEncoder(rowzip)
	reader := bufio.NewReaderSize(conn, 256*1024)
	header := make([]byte, 4)
	payload := make([]byte, 0, 256*1024)
	count := 0
	var lastTime, lastGameTime float64
	for {
		conn.SetReadDeadline(time.Now().Add(30 * time.Second))
		_, err = io.ReadFull(reader, header)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		size := int(binary.LittleEndian.Uint32(header))
		if size < 1 || size > 32*1024*1024 {
			return nil, fmt.Errorf("invalid protobuf size %d", size)
		}
		if cap(payload) < size {
			payload = make([]byte, size)
		} else {
			payload = payload[:size]
		}
		if _, err = io.ReadFull(reader, payload); err != nil {
			return nil, err
		}
		if _, err = compressed.Write(header); err != nil {
			return nil, err
		}
		if _, err = compressed.Write(payload); err != nil {
			return nil, err
		}
		state := new(pb.CMsgBotWorldState)
		if err = proto.Unmarshal(payload, state); err != nil {
			return nil, err
		}
		lastTime, lastGameTime = float64(state.GetDotaTime()), float64(state.GetGameTime())
		for _, unit := range state.GetUnits() {
			if unit.GetName() == "npc_dota_hero_nevermore" && unit.GetTeamId() == uint32(team) {
				p := unit.GetLocation()
				row := selfRow{float64(state.GetDotaTime()), float64(state.GetGameTime()), state.GetGameState(), uint32(team),
					unit.GetHandle(), float64(p.GetX()), float64(p.GetY()), unit.GetHealth(), unit.GetMana(), unit.GetActionType()}
				if err = encoder.Encode(row); err != nil {
					return nil, err
				}
				break
			}
		}
		count++
	}
	return map[string]any{"team": team, "packets": count, "last_dota_time": lastTime, "last_game_time": lastGameTime}, nil
}

// EOF alone does not establish that both perspectives were captured to shutdown.
func validateNativeEnds(sides []map[string]any) error {
	if len(sides) != 2 {
		return fmt.Errorf("expected two completed native streams, got %d", len(sides))
	}
	for _, side := range sides {
		if side["packets"].(int) == 0 {
			return fmt.Errorf("empty native stream")
		}
	}
	if math.Abs(sides[0]["last_game_time"].(float64)-sides[1]["last_game_time"].(float64)) > .2 {
		return fmt.Errorf("native streams ended at different game times; capture is incomplete")
	}
	return nil
}

func main() {
	runtime.GOMAXPROCS(1)
	run := flag.String("run", "", "output directory")
	radiant := flag.Int("radiant", 12250, "native TCP port")
	dire := flag.Int("dire", 12251, "native TCP port")
	logPath := flag.String("log", "", "collect Lua column telemetry from this log")
	schemaPath := flag.String("schema", "", "column schema for Lua telemetry")
	hz := flag.Int("hz", 15, "policy decisions per game second")
	until := flag.Float64("until", 600, "diagnostic cut or full match limit")
	timeout := flag.Float64("timeout", 600, "maximum wall seconds")
	duelMode := flag.Bool("duel", false, "experimental native observation -> Go teacher -> scheduled Lua executor")
	bots := flag.String("bots", "", "isolated executor script directory")
	period := flag.Int("period", 4, "decision period in engine ticks")
	offset := flag.Int("offset", 4, "fixed action delay in engine ticks")
	allowLate := flag.Bool("allow-late", false, "execute newest due command even after its scheduled tick")
	policyAddress := flag.String("policy-address", "", "optional protobuf inference endpoint")
	policyTeam := flag.Int("policy-team", 0, "model-controlled team, 2 or 3")
	flag.Parse()
	if *run == "" {
		panic("--run is required")
	}
	if err := os.MkdirAll(*run, 0755); err != nil {
		panic(err)
	}
	if *duelMode {
		if *bots == "" || *logPath == "" {
			panic("duel mode requires --bots and --log")
		}
		if err := duel(*run, *bots, *logPath, [2]int{*radiant, *dire}, *period, *offset, *timeout, *until, *policyAddress, *policyTeam, *allowLate); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *logPath != "" {
		if err := record(*run, *logPath, *schemaPath, *hz, *until, *timeout); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	var wait sync.WaitGroup
	reports := make(chan map[string]any, 2)
	failures := make(chan error, 2)
	for team, port := range map[int]int{2: *radiant, 3: *dire} {
		wait.Add(1)
		go func(team, port int) {
			defer wait.Done()
			result, err := receive(*run, team, port)
			if err != nil {
				failures <- err
			} else {
				reports <- result
			}
		}(team, port)
	}
	wait.Wait()
	close(reports)
	close(failures)
	result := map[string]any{"training_started": false}
	sides := []map[string]any{}
	errors := []string{}
	for report := range reports {
		sides = append(sides, report)
	}
	for err := range failures {
		errors = append(errors, err.Error())
	}
	if err := validateNativeEnds(sides); err != nil {
		errors = append(errors, err.Error())
	}
	result["sides"] = sides
	result["errors"] = errors
	json.NewEncoder(os.Stdout).Encode(result)
	if len(errors) > 0 {
		os.Exit(1)
	}
}
