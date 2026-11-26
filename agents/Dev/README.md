# Dev bot

A C++ MCTS bot, run by the Python harness through `DevAgent.py`.

Run these from the repository root (`COMP34111-Hex`).

## 1. Build

```bash
bash agents/Dev/build.sh
```

Build again after every C++ change. Inside Docker, build there too: a Mac build does not run on Linux.

## 2. Play a game

```bash
python3 Hex.py -p1 "agents.Dev.DevAgent DevAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent"
```

Swap `-p1` and `-p2` to play as Blue. The bot plays with the network in `EvalNetwork/weights/best.bin`.
