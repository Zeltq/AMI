"""Outcome-independent transport admission for complete recorded matches."""
import json
from pathlib import Path
from ..native_data import records


def audit_match(match, output=True, prefix_seconds=None):
    run=Path(match['run']);decisions={};acks={};last={};issues=[];late8={2:0,3:0};count={2:0,3:0};maximum={2:0,3:0};terminal_pending=0
    expected_cut=prefix_seconds is not None and match.get('reason')=='diagnostic_cut'and match.get('time')==prefix_seconds
    if match.get('error') or (match.get('reason') not in ('death','tower','last_hits')and not expected_cut):
        issues.append('not_a_complete_match')
    for row in records(run/'native-decisions.pb.gz'):
        if row.HasField('decision'):
            d=row.decision;key=(d.team,d.sequence)
            if key in decisions:issues.append('duplicate_decision')
            expected=last.get(d.team)
            if expected is None:
                if d.sequence!=1:issues.append('missing_initial_decision')
            elif d.sequence!=expected[0]+1 or d.observed_tick!=expected[1]+4:issues.append('observation_sequence_gap')
            if row.observation.team_id!=d.team or d.execute_tick-d.observed_tick!=6:issues.append('invalid_clock_or_team')
            last[d.team]=(d.sequence,d.observed_tick)
            decisions[key]=(d.observed_tick,d.execute_tick,d.observed_dota_time)
        if row.HasField('execution'):
            e=row.execution;key=(e.team,e.sequence)
            if key in acks:issues.append('duplicate_ack')
            acks[key]=e.tick
    for (team,sequence),(observed,scheduled,dota_time) in decisions.items():
        actual=acks.get((team,sequence))
        if actual is None:
            # The match can end while a command is still inside its allowed
            # execution window. This is censored, not evidence of packet loss.
            if dota_time+11/30>=match['time']-.001:terminal_pending+=1
            else:issues.append('unexplained_missing_ack')
            continue
        delay=actual-observed;count[team]+=1;maximum[team]=max(maximum[team],delay);late8[team]+=delay>8
        if actual<scheduled:issues.append('early_execution')
        if delay>=12:issues.append('execution_at_T12_or_later')
    if any(key not in decisions for key in acks):issues.append('ack_without_decision')
    for team in (2,3):
        if not count[team]:issues.append('missing_team')
        elif late8[team]/count[team]>.001:issues.append('more_than_0.1_percent_after_T8')
    report={'passed':not issues,'issues':sorted(set(issues)),'acknowledged':count,'after_T8':late8,'max_delay_ticks':maximum,'terminal_pending':terminal_pending,'terminal_pending_definition':'No ACK, but terminal falls within allowed T+11 execution window','scope':'bounded diagnostic prefix'if prefix_seconds is not None else'complete match','expected_prefix_seconds':prefix_seconds,
            'limits':{'nominal':6,'hard_reject_at':12,'late_after':8,'max_late_fraction':.001}}
    if output:(run/'transport-admission.json').write_text(json.dumps(report,indent=2))
    return report
