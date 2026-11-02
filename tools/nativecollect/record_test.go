package main

import "testing"

func pointer(value int) *int { return &value }

func TestFirstDeathAndSameTickTie(t *testing.T) {
	r := &recorder{Events: []terminal{{Team: 2, Time: 19, Reason: "death"}}}
	if result := r.result(); result.Winner == nil || *result.Winner != 3 || result.Time != 19 {
		t.Fatal(result)
	}
	r.Events = append(r.Events, terminal{Team: 3, Time: 19, Reason: "death"})
	if r.result().Winner != nil {
		t.Fatal("same-tick deaths must draw")
	}
}

func TestTimeoutUsesLastHits(t *testing.T) {
	r := &recorder{Events: []terminal{{Team: 2, Time: 600, Reason: "timeout", LH: 55}, {Team: 3, Time: 600, Reason: "timeout", LH: 56}}}
	result := r.result()
	if result.Winner == nil || *result.Winner != 3 || result.Scores["2"] != 55 {
		t.Fatal(result)
	}
	r.Events[1].LH = 55
	if r.result().Winner != nil {
		t.Fatal("equal last hits must draw")
	}
}

func TestReadyWaitsForOtherActorClock(t *testing.T) {
	r := &recorder{Events: []terminal{{Team: 2, Time: 19, Reason: "death"}}, Last: map[int]*frame{3: {Time: 19}}}
	result := &outcome{Reason: "death", Time: 19, Winner: pointer(3)}
	if r.ready(result) {
		t.Fatal("other side can still report a same-tick death")
	}
	r.Last[3].Time = 19.06667
	if !r.ready(result) {
		t.Fatal("later actor clock should complete the referee result")
	}
}

func TestNativeEarlyEOFMustFail(t *testing.T) {
	sides := []map[string]any{{"packets": 100, "last_game_time": 10.0}, {"packets": 1000, "last_game_time": 100.0}}
	if validateNativeEnds(sides) == nil {
		t.Fatal("early EOF accepted")
	}
	sides[0]["last_game_time"] = 99.94
	if err := validateNativeEnds(sides); err != nil {
		t.Fatal(err)
	}
	sides[0]["packets"] = 0
	if validateNativeEnds(sides) == nil {
		t.Fatal("empty stream accepted")
	}
}
