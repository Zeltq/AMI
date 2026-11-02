// Validate a human-review replay. This does not extract training examples.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/dotabuff/manta"
	"github.com/dotabuff/manta/dota"
)

func main() {
	if len(os.Args) < 2 || len(os.Args) > 3 {
		panic("usage: replaycheck path.dem [local-game-build]")
	}
	file, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer file.Close()
	parser, err := manta.NewStreamParser(file)
	if err != nil {
		panic(err)
	}
	if len(os.Args) == 3 {
		build, err := strconv.ParseUint(os.Args[2], 10, 32)
		if err != nil {
			panic(err)
		}
		parser.GameBuild = uint32(build)
	}
	result := map[string]interface{}{"path": os.Args[1], "training_data_extracted": false}
	packets := 0
	combat := map[string]map[string]uint64{}
	casts := map[string]uint64{}
	parser.Callbacks.OnCMsgDOTACombatLogEntry(func(entry *dota.CMsgDOTACombatLogEntry) error {
		attacker, _ := parser.LookupStringByIndex("CombatLogNames", int32(entry.GetAttackerName()))
		if attacker != "npc_dota_hero_nevermore" {
			return nil
		}
		key := strconv.FormatUint(uint64(entry.GetAttackerTeam()), 10)
		if combat[key] == nil {
			combat[key] = map[string]uint64{}
		}
		side := combat[key]
		inflictor, _ := parser.LookupStringByIndex("CombatLogNames", int32(entry.GetInflictorName()))
		target, _ := parser.LookupStringByIndex("CombatLogNames", int32(entry.GetTargetName()))
		if entry.GetType() == dota.DOTA_COMBATLOG_TYPES_DOTA_COMBATLOG_ABILITY {
			casts[inflictor]++
		}
		if entry.GetType() == dota.DOTA_COMBATLOG_TYPES_DOTA_COMBATLOG_DAMAGE && target == "npc_dota_hero_nevermore" {
			side["damage_hits_by_inflictor:"+inflictor]++
			if strings.HasPrefix(inflictor, "nevermore_shadowraze") {
				side["raze_hits_on_enemy_sf"]++
				side["raze_damage_on_enemy_sf"] += uint64(entry.GetValue())
			}
		}
		return nil
	})
	parser.Callbacks.OnCDemoFileHeader(func(header *dota.CDemoFileHeader) error {
		result["header"] = header
		return nil
	})
	parser.Callbacks.OnCDemoFileInfo(func(info *dota.CDemoFileInfo) error {
		result["file_info"] = info
		return nil
	})
	parser.Callbacks.OnCDemoPacket(func(packet *dota.CDemoPacket) error {
		packets++
		return nil
	})
	parser.Callbacks.OnCDemoStop(func(stop *dota.CDemoStop) error {
		result["stop_record_found"] = true
		return nil
	})
	if err := parser.Start(); err != nil {
		panic(err)
	}
	result["last_tick"] = parser.Tick
	result["packets"] = packets
	result["parsed"] = true
	for team, counts := range combat {
		if len(counts) == 0 {
			delete(combat, team)
		}
	}
	result["sf_combat_log"] = combat
	result["sf_casts_both_teams"] = casts
	output, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(output))
}
