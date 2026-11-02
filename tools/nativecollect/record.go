package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type command struct {
	Match      string  `json:"match"`
	Seq        int     `json:"seq"`
	Observed   float64 `json:"observed_time"`
	Action     int     `json:"action"`
	Buy        int     `json:"buy"`
	Item       int     `json:"item"`
	X          int     `json:"x"`
	Y          int     `json:"y"`
	Target     int     `json:"target_id"`
	TargetType string  `json:"target_type"`
}
type frame struct {
	Codec   string            `json:"codec"`
	Version int               `json:"version"`
	Match   string            `json:"match"`
	Player  int               `json:"player"`
	Seq     int               `json:"seq"`
	Team    int               `json:"team"`
	Time    float64           `json:"time"`
	Self    []json.RawMessage `json:"self"`
	Enemy   []json.RawMessage `json:"enemy"`
	Creeps  json.RawMessage   `json:"creeps"`
	Towers  json.RawMessage   `json:"towers"`
	Trees   []struct {
		ID   int    `json:"id"`
		Type string `json:"type"`
	} `json:"trees"`
	Base struct {
		ID int  `json:"id"`
		TP bool `json:"tp"`
	} `json:"base"`
	Legal struct {
		Actions []bool `json:"actions"`
		Items   []bool `json:"items"`
		Buys    []bool `json:"buys"`
	} `json:"legal"`
	Probe struct {
		Heroes []struct {
			Name string `json:"name"`
		} `json:"allied_heroes"`
		Inventory [][]json.RawMessage `json:"inventory"`
		Skills    [][]json.RawMessage `json:"skill_slots"`
	} `json:"probe"`
	Errors json.RawMessage `json:"errors"`
	Demo   struct {
		Command   command `json:"command"`
		Tick      int     `json:"decision_tick"`
		Period    int     `json:"period_ticks"`
		Issued    int     `json:"issued_hero_orders"`
		Execution struct {
			Seq      int  `json:"seq"`
			Accepted bool `json:"accepted"`
		} `json:"execution"`
		Receipts []struct {
			Name   string `json:"name"`
			Result int    `json:"result"`
		} `json:"purchase_receipts"`
	} `json:"demonstration"`
}
type terminal struct {
	Match  string  `json:"match"`
	Team   int     `json:"team"`
	Time   float64 `json:"time"`
	Reason string  `json:"reason"`
	LH     int     `json:"last_hits"`
}
type outcome struct {
	Reason string         `json:"reason"`
	Time   float64        `json:"time"`
	Winner *int           `json:"winner"`
	Scores map[string]int `json:"scores,omitempty"`
}
type bufferedFrame struct {
	Raw   []byte
	Frame *frame
}
type fragment struct {
	Count int
	Parts map[int][]byte
}
type recorder struct {
	Columns map[string]int
	Hz      int
	Match   string
	Last    map[int]*frame
	Events  []terminal
	Pending []bufferedFrame
	Output  *gzip.Writer
	Frames  map[string]int
	Actions map[string]int
	Buys    map[string]int
}

func number(row []json.RawMessage, index int) float64 {
	if index < 0 || index >= len(row) {
		return math.NaN()
	}
	var value float64
	if json.Unmarshal(row[index], &value) != nil {
		return math.NaN()
	}
	return value
}
func textAt(row []json.RawMessage, index int) string {
	if index < 0 || index >= len(row) {
		return ""
	}
	var value string
	json.Unmarshal(row[index], &value)
	return value
}
func boolAt(row []json.RawMessage, index int) bool {
	if index < 0 || index >= len(row) {
		return false
	}
	var value bool
	json.Unmarshal(row[index], &value)
	return value
}
func allowedItem(name string) bool {
	switch name {
	case "item_wraith_band", "item_boots", "item_bracer", "item_tpscroll", "item_enchanted_mango",
		"item_flask", "item_tango", "item_clarity", "item_circlet", "item_slippers", "item_gauntlets",
		"item_recipe_bracer", "item_recipe_wraith_band":
		return true
	}
	return false
}
func (r *recorder) validate(f *frame) error {
	c := f.Demo.Command
	if f.Codec != "sf-columns-1" || f.Version != 1 || (f.Team != 2 && f.Team != 3) {
		return fmt.Errorf("wrong frame contract")
	}
	if r.Match == "" {
		r.Match = f.Match
	}
	if f.Match != r.Match || c.Match != f.Match || c.Seq != f.Seq || math.Abs(c.Observed-f.Time) > .00001 {
		return fmt.Errorf("decision identity/time mismatch")
	}
	if f.Demo.Period < 1 || 30/f.Demo.Period != r.Hz || f.Demo.Tick%f.Demo.Period != 0 || math.Abs(float64(f.Demo.Tick)/30-f.Time) > .002 {
		return fmt.Errorf("decision clock mismatch")
	}
	if last := r.Last[f.Team]; last != nil && (last.Seq+1 != f.Seq || last.Demo.Tick+f.Demo.Period != f.Demo.Tick) {
		return fmt.Errorf("decision gap team %d", f.Team)
	}
	if !f.Demo.Execution.Accepted || f.Demo.Execution.Seq != f.Seq {
		return fmt.Errorf("rejected execution")
	}
	if len(f.Self) != len(r.Columns) || number(f.Self, r.Columns["health"]) <= 0 {
		return fmt.Errorf("invalid controlled hero")
	}
	if len(f.Legal.Actions) != 11 || len(f.Legal.Items) != 8 || len(f.Legal.Buys) != 7 || c.Action < 0 || c.Action >= 11 || c.Buy < 0 || c.Buy >= 7 || !f.Legal.Actions[c.Action] || !f.Legal.Buys[c.Buy] {
		return fmt.Errorf("masked action/purchase")
	}
	if c.X < 0 || c.X > 128 || c.Y < 0 || c.Y > 128 {
		return fmt.Errorf("coordinates outside policy space")
	}
	expected := 1
	if c.Action == 0 {
		expected = 0
	}
	if f.Demo.Issued != expected {
		return fmt.Errorf("unexpected executed order count")
	}
	if c.Action == 10 && (c.Item < 0 || c.Item >= 8 || !f.Legal.Items[c.Item]) {
		return fmt.Errorf("masked item")
	}
	if (c.Buy == 0) != (len(f.Demo.Receipts) == 0) {
		return fmt.Errorf("purchase receipt mismatch")
	}
	for _, receipt := range f.Demo.Receipts {
		if receipt.Result != -1 || !allowedItem(receipt.Name) {
			return fmt.Errorf("failed/forbidden purchase")
		}
	}
	if len(f.Probe.Heroes) != 1 || f.Probe.Heroes[0].Name != "npc_dota_hero_nevermore" {
		return fmt.Errorf("extra hero")
	}
	for _, row := range f.Probe.Inventory {
		if len(row) != 2 || !allowedItem(textAt(row, 1)) {
			return fmt.Errorf("forbidden inventory")
		}
	}
	for _, row := range f.Probe.Skills {
		if len(row) != 3 {
			return fmt.Errorf("invalid skill row")
		}
		name := textAt(row, 1)
		if (name == "nevermore_frenzy" || name == "special_bonus_attributes") && number(row, 2) != 0 {
			return fmt.Errorf("forbidden skill")
		}
	}
	errText := strings.TrimSpace(string(f.Errors))
	if errText != "" && errText != "[]" && errText != "{}" && errText != "null" {
		return fmt.Errorf("Bot API error: %s", errText)
	}
	targeted := c.Action == 3 || c.Action == 10 && (c.Item == 3 || c.Item == 6)
	if targeted {
		found := 0
		check := func(row []json.RawMessage) {
			if len(row) != len(r.Columns) || number(row, r.Columns["id"]) != float64(c.Target) {
				return
			}
			if c.TargetType == "tree" {
				return
			}
			if c.Action == 3 && boolAt(row, r.Columns["attackable"]) || c.Action == 10 && c.Item == 3 && boolAt(row, r.Columns["tp"]) {
				found++
			}
		}
		check(f.Enemy)
		for _, group := range []json.RawMessage{f.Creeps, f.Towers} {
			var rows [][]json.RawMessage
			if err := json.Unmarshal(group, &rows); err != nil {
				return err
			}
			for _, row := range rows {
				check(row)
			}
		}
		if c.Action == 10 && c.Item == 6 && c.TargetType == "tree" {
			for _, tree := range f.Trees {
				if tree.ID == c.Target && tree.Type == "tree" {
					found++
				}
			}
		}
		if c.Action == 10 && c.Item == 3 && f.Base.ID == c.Target && f.Base.TP {
			found++
		}
		if found != 1 {
			return fmt.Errorf("target absent/ambiguous/masked: %d", c.Target)
		}
	}
	r.Last[f.Team] = f
	return nil
}
func (r *recorder) write(f bufferedFrame) error {
	if _, err := r.Output.Write(f.Raw); err != nil {
		return err
	}
	if _, err := r.Output.Write([]byte{'\n'}); err != nil {
		return err
	}
	r.Frames[strconv.Itoa(f.Frame.Team)]++
	r.Actions[strconv.Itoa(f.Frame.Demo.Command.Action)]++
	r.Buys[strconv.Itoa(f.Frame.Demo.Command.Buy)]++
	return nil
}
func (r *recorder) flush(cutoff float64) error {
	keep := r.Pending[:0]
	for _, f := range r.Pending {
		if f.Frame.Time < cutoff-.001 {
			if err := r.write(f); err != nil {
				return err
			}
		} else {
			keep = append(keep, f)
		}
	}
	r.Pending = keep
	return nil
}
func (r *recorder) result() *outcome {
	var first *terminal
	for i := range r.Events {
		e := &r.Events[i]
		if e.Reason != "timeout" && (first == nil || e.Time < first.Time) {
			first = e
		}
	}
	if first != nil {
		losers := map[int]bool{}
		for _, e := range r.Events {
			if e.Reason != "timeout" && math.Abs(e.Time-first.Time) < .001 {
				losers[e.Team] = true
			}
		}
		result := &outcome{Reason: first.Reason, Time: first.Time}
		if len(losers) == 1 {
			winner := 5 - first.Team
			result.Winner = &winner
		}
		return result
	}
	scores := map[string]int{}
	for _, e := range r.Events {
		if e.Reason == "timeout" {
			scores[strconv.Itoa(e.Team)] = e.LH
		}
	}
	if len(scores) != 2 {
		return nil
	}
	result := &outcome{Reason: "last_hits", Time: 600, Scores: scores}
	if scores["2"] != scores["3"] {
		winner := 2
		if scores["3"] > scores["2"] {
			winner = 3
		}
		result.Winner = &winner
	}
	return result
}
func (r *recorder) ready(o *outcome) bool {
	if o == nil {
		return false
	}
	if o.Reason == "last_hits" {
		return true
	}
	for _, team := range []int{2, 3} {
		seen := false
		for _, e := range r.Events {
			if e.Team == team && e.Time >= o.Time-.001 {
				seen = true
			}
		}
		if f := r.Last[team]; f != nil && f.Time > o.Time+.001 {
			seen = true
		}
		if !seen {
			return false
		}
	}
	return true
}

func record(run, logPath, schemaPath string, hz int, until, timeout float64) (finalErr error) {
	schemaData, err := os.ReadFile(schemaPath)
	if err != nil {
		return err
	}
	var schema struct {
		Unit []string `json:"unit"`
	}
	if err = json.Unmarshal(schemaData, &schema); err != nil {
		return err
	}
	outputPath := filepath.Join(run, "frames.wire.jsonl.gz")
	output, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	zip, _ := gzip.NewWriterLevel(output, gzip.BestSpeed)
	r := &recorder{Columns: map[string]int{}, Hz: hz, Last: map[int]*frame{}, Output: zip,
		Frames: map[string]int{}, Actions: map[string]int{}, Buys: map[string]int{}}
	for i, name := range schema.Unit {
		r.Columns[name] = i
	}
	var result *outcome
	defer func() {
		if result != nil {
			if err := r.flush(result.Time); err != nil && finalErr == nil {
				finalErr = err
			}
		}
		if err := zip.Close(); err != nil && finalErr == nil {
			finalErr = err
		}
		output.Close()
		hash := sha256.New()
		file, err := os.Open(outputPath)
		if err == nil {
			io.Copy(hash, file)
			file.Close()
		}
		info, _ := os.Stat(outputPath)
		errors := []string{}
		if finalErr != nil {
			errors = append(errors, finalErr.Error())
		}
		sides := map[string]map[string]float64{}
		total := 0
		for _, n := range r.Frames {
			total += n
		}
		for team, f := range r.Last {
			values := map[string]float64{}
			for _, key := range []string{"last_hits", "denies", "level", "health"} {
				values[key] = number(f.Self, r.Columns[key])
			}
			sides[strconv.Itoa(team)] = values
		}
		report := map[string]any{"outcome": result, "errors": errors, "decision_hz": hz, "sides": sides,
			"eligible": finalErr == nil && result != nil && result.Reason != "diagnostic_cut", "training_started": false,
			"format_version": 2, "spec_version": 1, "data": map[string]any{"file": "frames.wire.jsonl.gz",
				"frames": total, "perspectives": r.Frames, "actions": r.Actions, "purchases": r.Buys,
				"bytes": info.Size(), "sha256": hex.EncodeToString(hash.Sum(nil))}}
		data, _ := json.MarshalIndent(report, "", "  ")
		os.WriteFile(filepath.Join(run, "go-result.tmp"), data, 0644)
		os.Rename(filepath.Join(run, "go-result.tmp"), filepath.Join(run, "go-result.json"))
	}()
	log, err := os.Open(logPath)
	if err != nil {
		return err
	}
	defer log.Close()
	reader := bufio.NewReaderSize(log, 256*1024)
	partial := []byte{}
	started := false
	fragments := map[[2]int]*fragment{}
	deadline := time.Now().Add(time.Duration(timeout * float64(time.Second)))
	for time.Now().Before(deadline) {
		line, readErr := reader.ReadBytes('\n')
		partial = append(partial, line...)
		if readErr == io.EOF {
			time.Sleep(5 * time.Millisecond)
			continue
		}
		if readErr != nil {
			return readErr
		}
		line = bytes.TrimSuffix(bytes.TrimSuffix(partial, []byte{'\n'}), []byte{'\r'})
		partial = nil
		if bytes.Contains(line, []byte("SF_CAPTURE_READY")) {
			started = true
			fragments = map[[2]int]*fragment{}
			continue
		}
		if i := bytes.Index(line, []byte("SF_RULE_ERROR|")); i >= 0 {
			return fmt.Errorf("teacher: %s", line[i+14:])
		}
		if !started {
			continue
		}
		if i := bytes.Index(line, []byte("SF_END|")); i >= 0 {
			var e terminal
			if err = json.Unmarshal(line[i+7:], &e); err != nil {
				return err
			}
			if e.Match != r.Match || (e.Team != 2 && e.Team != 3) || (e.Reason != "death" && e.Reason != "tower" && e.Reason != "timeout") {
				return fmt.Errorf("invalid terminal event")
			}
			r.Events = append(r.Events, e)
		} else if i := bytes.Index(line, []byte("SF_OBS|")); i >= 0 {
			fields := bytes.SplitN(line[i:], []byte{'|'}, 6)
			if len(fields) != 6 {
				return fmt.Errorf("truncated telemetry header")
			}
			values := [4]int{}
			for j := range values {
				value, err := strconv.Atoi(string(fields[j+1]))
				if err != nil {
					return err
				}
				values[j] = value
			}
			pid, seq, part, count := values[0], values[1], values[2], values[3]
			if part < 1 || part > count || count > 128 {
				return fmt.Errorf("invalid fragment count")
			}
			key := [2]int{pid, seq}
			packet := fragments[key]
			if packet == nil {
				packet = &fragment{Count: count, Parts: map[int][]byte{}}
				fragments[key] = packet
			}
			if packet.Count != count || packet.Parts[part] != nil {
				return fmt.Errorf("duplicate/inconsistent fragment")
			}
			packet.Parts[part] = append([]byte(nil), fields[5]...)
			if len(packet.Parts) == count {
				body := []byte{}
				for j := 1; j <= count; j++ {
					body = append(body, packet.Parts[j]...)
				}
				delete(fragments, key)
				f := new(frame)
				if err = json.Unmarshal(body, f); err != nil {
					return err
				}
				if f.Player != pid || f.Seq != seq {
					return fmt.Errorf("fragment identity mismatch")
				}
				if err = r.validate(f); err != nil {
					os.WriteFile(filepath.Join(run, "rejected-wire.json"), body, 0644)
					return err
				}
				r.Pending = append(r.Pending, bufferedFrame{body, f})
				if a, b := r.Last[2], r.Last[3]; a != nil && b != nil {
					if err = r.flush(math.Min(a.Time, b.Time) - float64(a.Demo.Period)/30); err != nil {
						return err
					}
				}
			}
			if len(fragments) > 8 {
				return fmt.Errorf("incomplete packets accumulating")
			}
		}
		candidate := r.result()
		if r.ready(candidate) {
			result = candidate
			return nil
		}
		if until < 600 {
			if a, b := r.Last[2], r.Last[3]; a != nil && b != nil && math.Min(a.Time, b.Time) >= until {
				result = &outcome{Reason: "diagnostic_cut", Time: until}
				return nil
			}
		}
	}
	return fmt.Errorf("Go recorder timeout")
}
