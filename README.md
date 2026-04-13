# STS2 MuZero Prototype

This directory contains a shell-first MuZero-style reinforcement learning prototype for Slay the Spire 2.

It reuses the singleplayer state/action semantics from `repos/STS2MCP` and talks to the live game through the same localhost API exposed by the mod.

The current version can also auto-dismiss post-run screens and start a fresh run with a preset character, so training can continue without manual menu clicks.

## Scope

- singleplayer only
- shell runtime only
- no external Python dependencies
- live environment over STS2MCP
- step-by-step training logs printed to the shell

## Included pieces

- `sts2_muzero/bridge.py`
  - mirrors the STS2MCP singleplayer tools as Python methods
- `sts2_muzero/env.py`
  - polls game state
  - builds legal actions
  - encodes observations
  - shapes rewards
- `sts2_muzero/muzero.py`
  - replay buffer
  - minimal MuZero-style representation/dynamics/prediction model
  - shallow root search
  - episode-end SGD training
  - boundary-weighted replay sampling for combat / act / run endings
- `sts2_muzero/cli.py`
  - shell training loop
  - combat / act / run summary logs

## Limitations

- This is a first working prototype, not a production MuZero implementation.
- Search is shallow because the live interface cannot clone game state.
- Training speed is bounded by the live game and shell logging.

## Run

1. Install and enable the STS2MCP mod in Slay the Spire 2.
2. Start the game.
3. Start or continue a singleplayer run.
4. Run:

```powershell
python run_training.py --resume
```

Example flags:

```powershell
python run_training.py --character Ironclad --ascension 0 --resume
python run_training.py --simulations 32 --updates-per-episode 4 --checkpoint-interval 25
python run_training.py --temperature 0.25 --no-root-noise
python run_training.py --max-steps 200
python run_training.py --manual-start
python run_training.py --enable-episode-logs
python run_training.py --enable-episode-logs --episode-log-dir episode_logs --episode-log-model-name latest
```

Example training configuration:

```powershell
python .\run_training.py `
  --max-steps 50000 `
  --simulations 40 `
  --combat-selection-simulations 10 `
  --combat-selection-search-time-budget-seconds 0.03 `
  --learning-rate 1.5e-4 `
  --discount 0.997 `
  --updates-per-episode 8 `
  --replay-capacity 20000 `
  --warmup-samples 1500 `
  --temperature 0.45 `
  --checkpoint-path checkpoints\ep_v1.json `
  --checkpoint-interval 250 `
  --character Ironclad `
  --ascension 0 `
  --floor-advance-weight 0.10 `
  --act-advance-weight 6.0 `
  --hp-delta-weight 5.0 `
  --gold-delta-weight 0.25 `
  --gold-delta-scale 25 `
  --enemy-hp-delta-weight 8.0 `
  --turn-end-weight 0.0 `
  --turn-skip-unspent-penalty 6.0 `
  --combat-end-weight 6.0 `
  --combat-defeat-weight -10.0 `
  --act-end-weight 2.0 `
  --run-victory-weight 12.0 `
  --run-defeat-weight -18.0 `
  --combat-tactical-shaping 0.30 `
  --end-turn-slack-penalty 0.0 `
  --episode-settlement-weight 0.15 `
  --episode-settlement-decay 0.995 `
  --settlement-signal-mode mean `
  --settlement-normalization sum `
  --settlement-clip 6 `
  --card-select-candidate-limit 6 `
  --end-turn-guard-timeout-seconds 2.5 `
  --card-reward-preview-guard-timeout-seconds 2.5 `
  --map-route-defeat-gold-threshold 250 `
  --map-route-defeat-gold-penalty-multiplier 2.0 `
  --enable-episode-logs `
  --map-route-choice-weight 1.5 `
  --resume
```

Episode log flags:

- `--enable-episode-logs`
  - enables compressed `.logs` archive writing; default is off
- `--disable-episode-logs` / `--no-episode-logs`
  - disables archive writing explicitly
- `--episode-log-dir episode_logs`
  - writes one compressed `.logs` archive per finished episode
- `--episode-log-model-name latest`
  - overrides the `model` segment in the filename

Each archive filename follows:

```text
<character>-<highest_floor>-<episode_index>-<date:YYYYMMDD>-<time:HHMMSS>-<model>.logs
```

The archive stores:

- one lossless global-state store for the whole episode, compressed with JSON delta patches plus `lzma/xz`
- one floor node per visited map node, grouped by node type rather than one file per combat
- one action timeline inside each node with only the changed global-state fields listed per step
- node-specific details for:
  - combat: battle entry snapshot, turn grouping for both sides, post-combat rewards, result
  - shop: shop inventory snapshot and purchase records
  - rest site: campfire options and chosen action
  - event: event options, selected option, dialogue progression, and resulting state deltas
  - question mark: original `?` route plus `resolved_type`, then the resolved node content
  - treasure: available rewards and claimed item
- final episode result, highest floor, node counts, and reward summary

The archive intentionally does not store MuZero search internals, observations, or model state. It stores only the environment-facing episode state/action/result data needed for offline replay or training data extraction.

Archives are written as minified JSON compressed with built-in `lzma` so they stay lossless and can be reconstructed exactly.

## Read Episode Logs

```powershell
python -m sts2_muzero.log_reader summary episode_logs\Ironclad-17-52-20260407-224430-latest.logs
python -m sts2_muzero.log_reader nodes episode_logs\...\some.logs
python -m sts2_muzero.log_reader timeline episode_logs\...\some.logs --node-index 3
python -m sts2_muzero.log_reader show-node episode_logs\...\some.logs 3 --states both
python -m sts2_muzero.log_reader show-step episode_logs\...\some.logs 3 5 --states both
python -m sts2_muzero.log_reader export-json episode_logs\...\some.logs --output expanded.json
```

If the package is installed, the same reader is also exposed as:

```powershell
sts2-muzero-logs summary <path-to-log>
```

Useful reader commands:

- `summary`
  - print episode metadata, reward summary, and node counts
- `nodes`
  - list all floor nodes with type, resolved type, route info, and node-specific counts
- `timeline`
  - print the action timeline for the whole episode or a single node
- `show-node`
  - dump one node in full, optionally with entry/final global state
- `show-step`
  - dump one action record in full, optionally with before/after global state
- `export-json`
  - export decompressed JSON with all decoded global states for downstream training or analysis

## Example shell output

```text
[09:12:03] step=18 ep=1 act=1 floor=4 hp=63/80 state=monster action=play Strike -> Jaw Worm reward=+1.240 ep_raw_reward=+8.510 root_value=+0.318 loss=deferred_ep
[09:12:17] step=26 ep=1 act=1 floor=4 hp=58/80 state=rewards action=end turn reward=+8.420 ep_raw_reward=+16.930 root_value=+0.502 loss=deferred_ep boundary=combat_end
[09:12:17] combat_end ep=1 act=1 floor=4 steps=9 raw_reward=+10.280 result=victory
[09:24:51] act_summary ep=1 act=1 reason=act_end steps=88 floors=1->18 combats=8 raw_reward=+54.720
[09:40:02] episode_train ep=1 updates=4 loss=0.8841 value=0.1932 reward=0.0510 policy=0.5319 consistency=0.1080
[09:40:02] run_end ep=1 steps=173 combats=17 raw_return=+71.580 episode_settlement=+3.240 result=defeat replay=173 combat=17 act=1 run=1
```
