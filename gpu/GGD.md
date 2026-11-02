# GGD imitation learning

Actual results of the first full collection and three completed IL epochs,
including the interrupted fourth epoch and known limitations:
[run summary, 2026-11-03](../docs/ggd0-il-20261103.md).

GGD uses the current Policy architecture with fresh random weights. The first
completed imitation checkpoint is GGD0. Subsequent authorized self-play stages
will produce GGD1, GGD2, etc. Short pipeline tests are named GGD0-smoke and are
not release models.

## Input change

`ggd-v2` appends combat inputs to the existing observation layout. Only the
first hero, unit and projectile linear layers grow: 96 -> 118, 96 -> 118 and
12 -> 28 inputs, respectively. This adds 4,736 parameters (1,146,029 total).
Entity pooling, terrain CNN, LSTM-256 and action heads retain their sizes.
The default constructor and the original MIA observation profile retain their
old dimensions. No MIA weights initialize GGD.

Each unit receives a visible attack target's relative position and HP, damage
spread, incoming attacker count/damage/DPS, incoming projectile count/damage,
and the next projectile's ETA/damage. Each projectile receives a visible
target's relative position/HP/team, last observed caster damage/type, and the
caster's current relative position when visible. Every field has a known flag.
Unit handles are used to build relations; opaque handle numbers are not network
features. Dead casters can retain their previously observed damage/type.

Both the GPU observer and the real Dota adapters build these features. The Dota
damage table/spread defaults currently match calibration 003; another engine
calibration requires updating/pinning the same constants in both adapters.
No future states, future RNG or rolled projectile damage become inputs.

The teacher still reads exact internal attack-ready and windup-release times
and target generations. GGD must infer attack timing from public observations
and recurrent history. The additions remove the missing target relations;
they do not prove that every teacher decision can be reconstructed exactly.
Real Dota observation parity and gameplay transfer remain to be measured.

## Recording

Run from `C:\AMI` after a release build:

```powershell
cargo build --release --locked --manifest-path gpu/Cargo.toml
gpu/target/release/sf-duel-gpu.exe collect-teacher gpu/configs/ggd-collect.json runs/gpu/ggd-demonstrations-001
python gpu/scripts/ggd_dataset.py runs/gpu/ggd-demonstrations-001
```

The supplied configuration requests 1,024 teacher-v9 games, 100 concurrent
arenas, up to 600 seconds after game time zero, plus the configured pregame.
The native terminal rule can finish an episode earlier. Only the first episode
in each arena is recorded. Both actors from a game share a split, determined by
game index modulo 8: 768 train, 128 validation, 128 test games. Test games are
never used by the trainer to choose a checkpoint.

The collector keeps simulation, observations, teacher decisions and action
delays on GPU. It copies compact FP16 observation sequences every 128 decisions
to a bounded writer, with original actions, life resets and validity flags.
It preserves all occupied entity buckets. Gzip files have SHA-256 hashes and a
versioned header. No samples are dropped when the writer is busy. The engine
configuration/assets and teacher/observer hashes are pinned in the dataset.
Keep `dataset.json`, `environment`, `config.json`, all block files and reports.

Decisions run every 4 ticks at 30 Hz. Commands use offset_ticks=6 and the
existing executor's due+1 check (first execution seven ticks after observation).
No teacher commands bypass the execution queue. `stop.json` requests a stop at
a block boundary; partial datasets are accepted only for smoke training.

Measured preparation run: 100 arenas, 1,024 decisions per actor, 204,800 valid
decisions in 39.442 s, 67,556,914 compressed bytes. This was a partial prefix,
not 100 completed games. A separate 10-arena 45-second prefix completed its
configured recording horizon, yielding 19,200 decisions / 113 attack labels
in 46.472 s. It completed zero native terminal games.

At the measured rate, 1,024 full-length recordings contain at most about
10.49 million decisions and take about 34 minutes by linear extrapolation.
Budget 35-70 minutes and 3-6 GB; later waves and cohort setup can change the
rate. These are estimates, not a completed full collection.

## GPU training

```powershell
gpu/target/release/sf-duel-gpu.exe train-il gpu/configs/ggd-il.json runs/gpu/GGD0
```

On this Windows machine the command invokes the project's WSL Ubuntu
`scripts/wsl_python.sh` and `gpu/scripts/train_ggd.py`. The existing PyTorch
Policy trains on ROCm; no CPU fallback is allowed. Vulkan/CubeCL still perform
collection and native inference. The preparation check used PyTorch
2.14.1+rocm7.2 and the AMD Radeon RX 9070 XT.

Training starts a fresh model and optimizer. The starting configuration is
5 epochs, 8 sequences x 128 decisions per batch, AdamW at 1e-4, weight decay
1e-5 and gradient clipping at 1.0. The conditional action log likelihood
includes buy/item/target/coordinates only where the existing contract uses
them. Attack frames have weight 4. This is an initial hyperparameter choice;
validation attack recall and actual gameplay must determine its adequacy.
The value head receives no training loss and remains unchanged.

Recurrent state resets at each 128-step sequence and at life resets. Native
inference must use the exported 128-step reset and FP16 observation contract.
Selection uses validation NLL, and reports also include action accuracy,
attack precision/recall and conditional target accuracy with their counts.
Overall accuracy is dominated by CONTINUE; it is not a gameplay score.

A measured 8-update ROCm smoke changed the action head (L2 delta 0.0328023)
with a nonzero supervised gradient and zero value-head delta. Steady GPU batch
work averaged 0.067 s for 1,024 frames. Allow roughly 15-25 minutes per epoch
including loading/decoding, or about 1-2 hours for five epochs, until a full
dataset gives a better measurement. The native Burn IL experiment had zero
action-head gradients and was not used for training; its diagnostic source
and logs remain in the preparation artifact directory.

Fresh output directories prevent accidental replacement. `--max-updates N`
is a short technical test and produces GGD0-smoke. `stop.json` stops training.
`best.pt`, `last.pt`, optimizer state, provenance, history and result reports
are saved. Interrupted checkpoints are not completed GGD0 releases.

## Inference and quality gates

```powershell
python gpu/scripts/weights.py --checkpoint runs/gpu/GGD0/best.pt --output runs/gpu/GGD0/burn
gpu/target/release/sf-duel-gpu.exe evaluate-ggd runs/gpu/dota-calibration-003/engine.json runs/gpu/GGD0/burn/weights runs/gpu/GGD0/evaluation --matches 100 --seconds 600
```

The evaluator balances GGD versus teacher by side, runs the normal delayed
action queue and reports GGD's own LH/deny, teacher LH/deny, both-side credited
creep deaths, accessible creep deaths, arrival and hero deaths per game.
Always report numerators and denominators; a strong teacher opponent can hide
a weak learner in a combined LH/deny score. Native terminal completions are
counted separately from configured evaluation prefixes. Large evaluations and
real Dota games have not been run for GGD. Validate simulator behavior before
attempting a small real Dota transfer check. PPO and self-play training belong
to a later authorized stage.

Preparation artifacts and exact commands/results:
`runs/gpu/ggd-preparation-20261102-190559`.

## Authorized GGD0 run with gameplay reports

`runs/gpu/ggd0-il-20261102-202637` contains the frozen executable and run
configuration for 1,024 recordings followed by five IL epochs. After every
epoch the runner evaluates 32 GGD-versus-teacher and 32 GGD-versus-GGD games,
with the same seed and a 600-second horizon. Both sides are reported. These
gameplay evaluations add runtime beyond the IL batch processing estimate.

To reproduce these checks in a new run, add the following `evaluation` field
to the IL JSON configuration after building the executable. Paths are relative
to the repository root; replace `dataset` with the newly collected directory.

```json
"evaluation": {
  "executable": "gpu/target/release/sf-duel-gpu.exe",
  "engine": "runs/gpu/dota-calibration-003/engine.json",
  "matches": 32,
  "seconds": 600,
  "seed": 20261103,
  "modes": ["teacher", "mirror"]
}
```

The supervisor additionally expects `pipeline.json`, `collect.json` and
`train.json` in its run directory and a local `runs/pilot-status.json` JSON
object. These generated files are excluded from Git; the direct collection
and training commands above do not require this supervisor state.

`dataset/teacher-summary.json` and `dataset/game-metrics.json` hold teacher
statistics. Each training epoch has a checkpoint, exported native weights,
`epoch-NNN/report.json`, and `epoch-NNN/vs-teacher/report.json` plus
`epoch-NNN/vs-mirror/report.json`. `GGD0/epoch-metrics.csv` compares all epochs.
Reports include LH/deny, all/accessible lane deaths with numerators and
denominators, first death time, mid arrival/missing arrivals, stalls/idle time,
level/HP/mana/gold/net worth/souls, command histograms and minute snapshots.
Completed casts, raze hits and command-caused attack cancellations are counted
inside the physics tick. Command counts and actual spell completions are
separate. Arrival/position samples have 4/30-second resolution; the first
death time has 1/30-second resolution. Exact engine state is used for telemetry
only; it is not passed to GGD.

`gpu/scripts/run_ggd.py RUN_DIRECTORY` performs collection then training once,
updates `runs/pilot-status.json`, and records stage processes, commands and
errors. A `started.json` marker prevents duplicate launches. The run's
`stop.json` requests a graceful stop. Full real Dota evaluation and PPO are
outside this run.

The user-requested pause after epoch 2 and both evaluations is saved in
runs/gpu/ggd0-il-20261102-202637/GGD0/paused-epoch-002.pt. It contains model
weights, AdamW state and 15,990 optimizer updates. The automatic loop had
already entered epoch 3 when it was stopped; those partial updates were
discarded, as recorded in pause-result.json. No collection is needed again.

Resume with gpu/scripts/run_ggd.py RUN_DIRECTORY --resume CHECKPOINT. The
runner requires a paused status, refuses an open Dota client and creates a
fresh GGD0-resumed output directory. The trainer validates dataset SHA-256
and training parameters, restores weights and optimizer, then runs epochs
3 through 5 with the existing evaluations. --output selects a fresh output
directory if a previous resume attempt already created the default one.
CPU checkpoint inspection is available through train_ggd.py CONFIG UNUSED_OUTPUT
--resume CHECKPOINT --check-resume; it does not launch training or evaluation.
