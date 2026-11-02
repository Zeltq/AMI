from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    gamma: float = 1.0
    gae_lambda: float = 1.0
    learning_rate: float = 1e-4
    value_warmup_lr: float = 3e-4
    value_warmup_epochs: int = 3
    ppo_epochs: int = 4
    clip: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .005
    gradient_norm: float = .5
    target_kl: float = .01
    stop_kl: float = .015
    kl_check_batches: int = 10
    burn_in: int = 32
    bptt: int = 128
    batch_sequences: int = 16
    workers: int = 4
    timescale: float = 2.0
    period_ticks: int = 4
    offset_ticks: int = 6
    allow_late: bool = True
    collection_minutes: float = 105.0
    nominal_matches: int = 64
    archive_fraction: float = .2
    evaluation_matches: int = 10
    teacher_win_threshold: int = 8
    teacher_streak: int = 2
    teacher_recheck_cycles: int = 5
    seed: int = 20261030
    gradient_accumulation: int = 1
    enforce_quality: bool = False
    fresh_memory: bool = False

    def validate(self):
        if self.gamma != 1 or self.gae_lambda not in (1., .95):
            raise ValueError('Approved gamma=1; GAE lambda is 1 or .95')
        if not 1 <= self.ppo_epochs <= 4 or self.burn_in < 0 or self.bptt < 1:
            raise ValueError('Invalid recurrent PPO configuration')
        if not 0 < self.target_kl < self.stop_kl or self.batch_sequences < 1:
            raise ValueError('Invalid KL/minibatch configuration')
        if not 1 <= self.workers <= 10 or self.period_ticks != 4 or self.offset_ticks < 4:
            raise ValueError('Invalid game clock')
        if not self.allow_late or not 0 < self.archive_fraction < 1 or not 0 < self.clip < 1:
            raise ValueError('Invalid action/opponent/PPO settings')
        positive=(self.learning_rate,self.value_warmup_lr,self.gradient_norm,self.collection_minutes,self.timescale)
        if any(not math.isfinite(x) or x<=0 for x in positive):raise ValueError('Invalid positive parameter')
        if not 1<=self.value_warmup_epochs<=10 or self.kl_check_batches<1:raise ValueError('Invalid warmup/KL cadence')
        if self.gradient_accumulation not in (1,2):raise ValueError('Only microbatch16/accumulation1 or2 is authorized')
        if self.value_coefficient<0 or self.entropy_coefficient<0:raise ValueError('Negative loss coefficient')
        return self

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    def write(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))
