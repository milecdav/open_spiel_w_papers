# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Save Progress Frequently

**Conversations can disconnect (DC) at any time without warning.** After completing any significant milestone (passing tests, fixing bugs, implementing features), **immediately update PROJECT_PROGRESS.md** with what was done and the current state. Do NOT wait until the end of a session. Write incremental updates as you go. A DC with unsaved progress means repeating work in the next session.

## Project Overview

This is a fork of DeepMind's OpenSpiel framework with custom research paper implementations in `open_spiel/papers_with_code/`. The main additions are:

- **Public State CFR (PS-CFR)**: Efficient CFR exploiting public state structure
- **Depth Limited Responses**: SE-CFR (Safe Exploitation CFR) and RNR (Restricted Nash Response) algorithms
- **Gadget Games**: Safe subgame re-solving (Burch, Johanson, Bowling 2014) and max-margin gadget (Moravcik et al. 2016 / DeepStack)
- **Continual Resolving (CDRNR/ABD)**: Depth-limited resolving with RNR, model-as-portfolio MVS

## Critical Rules

- **NEVER** call `source /etc/profile.d/modules.sh` — `ml` is already available in the shell. Calling it breaks the environment.
- **NEVER** use `tail` to truncate build output.
- **Always** use `-j8` for parallel builds.
- **Always** load modules before building or running executables.
- C++17 required, Clang compiler, uses abseil (absl) for flags/strings.

## Reading PDF Files

The `Read` tool cannot render PDFs on this system (no poppler-utils). Use PyPDF2 with the correct Python module:

```bash
ml Python/3.11.5-GCCcore-13.2.0
python3 -c "
import PyPDF2
reader = PyPDF2.PdfReader('path/to/file.pdf')
print(f'Total pages: {len(reader.pages)}')
for i in range(min(5, len(reader.pages))):  # adjust range as needed
    print(f'=== PAGE {i+1} ===')
    print(reader.pages[i].extract_text())
"
```

- **IMPORTANT**: PyPDF2 requires `Python/3.11.5-GCCcore-13.2.0` module (NOT the build Python 3.9.5)
- After reading PDFs, reload the build modules: `ml Clang/12.0.1-GCCcore-10.3.0 CMake/3.20.1-GCCcore-10.3.0 Python/3.9.5-GCCcore-10.3.0`
- Adjust the page range to read specific sections
- For large papers, read a few pages at a time to avoid excessive output

## Build System

### 1. Load modules (required before every build or run)
```bash
ml Clang/12.0.1-GCCcore-10.3.0 CMake/3.20.1-GCCcore-10.3.0 Python/3.9.5-GCCcore-10.3.0
```

### 2. Run cmake (only needed once or after CMakeLists.txt changes)
```bash
cd open_spiel/build
OPEN_SPIEL_BUILD_WITH_ACPC=ON OPEN_SPIEL_BUILD_WITH_LIBNOP=ON \
OPEN_SPIEL_BUILD_WITH_PAPERS=ON OPEN_SPIEL_BUILD_WITH_PYTHON=ON \
OPEN_SPIEL_BUILD_WITH_ORTOOLS=ON cmake ..
```

### 3. Build a target
```bash
make <target_name> -j8
```

### Key build flags
- `OPEN_SPIEL_BUILD_WITH_PAPERS=ON` — Required for papers_with_code implementations
- `OPEN_SPIEL_BUILD_WITH_ORTOOLS=ON` — Enables LP solving (for computing exact game values)
- `OPEN_SPIEL_BUILD_WITH_ACPC=ON` — ACPC poker protocol support
- `BUILD_TYPE=Testing|Debug|Release` — Optimization level (Testing is default)

### Build targets
| Target | Description |
|--------|-------------|
| `my_main` | PS-CFR main executable |
| `dlr_main` | Depth Limited Responses main executable |
| `continual_resolving_test` | Continual resolving / ABD tests (17 tests) |
| `resolving_gadget_test` | Safe gadget tests (Kuhn + Leduc + Goofspiel, safe vs unsafe) |
| `max_margin_gadget_test` | Max-margin gadget tests (Kuhn + Leduc + Goofspiel, safe vs max-margin) |
| `unsafe_subgame_test` | Unsafe subgame unit tests |
| `matrix_valued_states_test` | Matrix-valued states tests |
| `ses_gadget_test` | SES gadget tests (8 tests) |
| `ox_gadget_test` | OX-Search gadget tests (6 tests) |

### Full installation (with Python bindings)
```bash
./install_pscfr.sh
# or
OPEN_SPIEL_BUILD_WITH_PAPERS=ON ./install.sh
```

### Run all tests
```bash
cd open_spiel/build && ctest --output-on-failure
```

## Running Experiments

### PS-CFR
```bash
./open_spiel/build/papers_with_code/public_state_cfr/my_main
```

### DLR
```bash
# Vanilla CFR
./open_spiel/build/papers_with_code/depth_limited_responses/dlr_main --cfr --iterations=1000

# MCCFR (Monte Carlo CFR)
./open_spiel/build/papers_with_code/depth_limited_responses/dlr_main --mccfr --iterations=1000

# SE-CFR (Safe Exploitation CFR)
./open_spiel/build/papers_with_code/depth_limited_responses/dlr_main --secfr --iterations=1000

# RNR (Restricted Nash Response)
./open_spiel/build/papers_with_code/depth_limited_responses/dlr_main --rnr --iterations=1000 --target_player=0

# Specify game
./open_spiel/build/papers_with_code/depth_limited_responses/dlr_main --game=leduc_poker --cfr --iterations=1000
```

Results are saved to `results/dlr/`.

### Cluster (SLURM)
Scripts in `slurm_scripts/dlr/` and `slurm_scripts/pscfr/`.

## Architecture

### Core OpenSpiel (C++)
- `open_spiel/spiel.h` — Core game API (Game, State, Observer classes)
- `open_spiel/algorithms/` — Algorithm implementations (CFR, MCTS, best response, etc.)
- `open_spiel/games/` — Game implementations (Leduc poker, Liar's Dice, etc.)
- `open_spiel/pybind11/pyspiel.cc` — Python bindings

### Game Transforms (`open_spiel/game_transforms/`)
- `continual_resolving.h/cc` — CDRNR/ABD: `ContinualResolve`, `ResolveSubgames`, `GadgetPolicyWrapper`, `MVSOpponentPolicy`, `MVSModelEntryPolicy`
- `subgame_utils.h/cc` — Reusable decomposition utilities
- `resolving_gadget.h/cc` — Safe re-solving gadget (Burch et al. 2014)
- `max_margin_gadget.h/cc` — Max-margin gadget (Moravcik et al. 2016)
- `ses_gadget.h/cc` — SES gadget (Liu et al. 2022)
- `ox_gadget.h/cc` — OX-Search gadget (Ge et al. 2024)
- `unsafe_subgame.h/cc` — Unsafe re-solving
- `matrix_valued_states.h/cc` — MVS depth-limited games with subtree pure strategy enumeration and model-as-portfolio

### Subgame Utilities (`subgame_utils.h/cc`)

Data structures:
- `SubgameRoot` — state + info_state_string + reach_prob for a subgame root
- `SubgameDecomposition` — complete decomposition: game, trunk_info_states, grouped_subgames, reach_probs[2], cfvs[2], chance_reach

High-level decomposition:
- `DecomposeGameAtRound(game, policy, round)` — decompose at a round boundary (chance-node count)
- `DecomposeGameAtDepth(game, policy, depth)` — decompose at a depth boundary (player-action count)
- `BuildSubgameRoots(states, resolving_player, reach_probs)` — build SubgameRoot vector from states

State collection:
- `CollectStates`, `CollectStatesAtDepth`, `CollectStatesAtRound`
- `CollectInfoStateStringsBeforeDepth`, `CollectInfoStateStringsBeforeRound`
- `CollectSubgameInfoStatesPerPlayer`

Computation:
- `ComputeReachProbabilities`, `ComputeCounterfactualValuesAtStates`
- `GroupStatesByPublicObservation`

### Custom Research Code
- `open_spiel/papers_with_code/public_state_cfr/` — PS-CFR implementation (`my_main.cc`, `subgame.cc/h`)
- `open_spiel/papers_with_code/depth_limited_responses/` — DLR paper (`dlr_main.cc`, `secfr.h/cc`)

### Key Algorithm Classes
- `algorithms::CFRSolverBase` — Base class for CFR variants
- `algorithms::RNRSolver` — Restricted Nash Response solver
- `algorithms::TabularBestResponse` — Best response computation
- `algorithms::ortools::SequenceFormLpSpecification` — LP-based game solving

## Project Progress

See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for detailed progress, test results, bug fixes, and design notes.
