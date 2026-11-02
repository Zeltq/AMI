"""The referee consumes authoritative own-side events, never policy predictions."""
from dataclasses import dataclass,field

@dataclass
class Referee:
    events:list=field(default_factory=list)

    def add(self,event):
        if event['team'] not in (2,3) or event['reason'] not in ('death','tower','timeout'):
            raise ValueError('Invalid native terminal event')
        self.events.append(event)

    def result(self):
        if not self.events:return None
        decisive=[e for e in self.events if e['reason']!='timeout']
        if decisive:
            first=min(e['time'] for e in decisive)
            at=[e for e in decisive if abs(e['time']-first)<.001]
            losers={e['team'] for e in at}
            return {'reason':at[0]['reason'],'time':first,
                    'winner':None if len(losers)==2 else 5-next(iter(losers))}
        scores={e['team']:e['last_hits'] for e in self.events if e['reason']=='timeout'}
        if set(scores)!={2,3}:return None
        return {'reason':'last_hits','time':600.,'scores':scores,
                'winner':None if scores[2]==scores[3] else max(scores,key=scores.get)}

    def ready(self,clocks):
        candidate=self.result()
        if candidate is None:return False
        if candidate['reason']=='last_hits':return True
        first=candidate['time']
        # A packet can be read between the two bot callbacks of one engine tick.
        # Wait for each side's terminal event or a later native observation.
        return all(any(e['team']==team and e['time']>=first-.001 for e in self.events)
            or clocks.get(team,float('-inf'))>first+.001 for team in (2,3))
