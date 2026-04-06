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
  - online SGD training
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
python run_training.py --simulations 32 --updates-per-step 4 --checkpoint-interval 25
python run_training.py --temperature 0.25 --no-root-noise
python run_training.py --max-steps 200
python run_training.py --manual-start
```

## Example shell output

```text
[09:12:03] step=18 ep=1 act=1 floor=4 hp=63/80 state=monster action=play Strike -> Jaw Worm reward=+1.240 ep_reward=+8.510 root_value=+0.318 loss=0.9412 value=0.2115 reward=0.0321 policy=0.5943 consistency=0.1033
[09:12:17] step=26 ep=1 act=1 floor=4 hp=58/80 state=rewards action=end turn reward=+8.420 ep_reward=+16.930 root_value=+0.502 loss=0.8841 value=0.1932 reward=0.0510 policy=0.5319 consistency=0.1080 boundary=combat_end
[09:12:17] combat_end ep=1 act=1 floor=4 steps=9 reward=+10.280 result=victory
[09:24:51] act_summary ep=1 act=1 reason=act_end steps=88 floors=1->18 combats=8 reward=+54.720
[09:40:02] run_end ep=1 steps=173 combats=17 shaped_return=+71.580 result=defeat replay=173 combat=17 act=1 run=1
```
