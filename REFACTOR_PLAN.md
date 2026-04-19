# Refactoring plan: Unified subgame-resolving API

**Status:** plan — not yet implemented.
**Author:** handoff for a weaker model (Sonnet) to execute.
**Scope:** `open_spiel/game_transforms/` plus a few callers in `papers_with_code/`.

## 0. Purpose

Today each gadget (`resolving_gadget`, `max_margin_gadget`, `ses_gadget`,
`ox_gadget`, `full_gadget`, `unsafe_subgame`) is its own end-to-end artifact,
and `continual_resolving.cc::ResolveSubgames` hand-dispatches on
`(SolverType, GadgetType)` with a giant `if/else` chain. SES is essentially
"max-margin gadget + RNR + don't lock the opponent in the subgame"; OX is a
similar recombination of a different gadget primitive. The duplication makes
it hard to mix and match.

Goal: factor subgame resolving into **four orthogonal axes**, any combination
of which should work without touching the dispatch code:

1. **Gadget** — how the subgame entry is wrapped in the **free** branch
2. **Response** — `Nash` / `RNR` (what kind of response we want)
3. **Solver** — `CFR` / `LP` (how we solve the resulting game; independent
   of the response type)
4. **LockOpponentInFixedBranch** — RNR-only flag: inside the fixed branch's
   unsafe subgame, is the opponent a chance node drawn from σ^fix (lock=true)
   or a free player with branch-local info states (lock=false)?

The key insight driving the Response/Solver split: RNR is an
opponent-robust response *concept*; CFR and LP are solution *methods*. Today
they're conflated because the existing `RNRSolver` is a CFR variant that bakes
`p` into its recursion, so LP+RNR isn't possible. We fix this by introducing a
**game transform** that physically encodes the `p`-mixture as a root chance
node splitting the game into two branches (§4.5), making RNR solver-agnostic.
After the transform any standard solver — CFR or LP — recovers the RNR response.

**Architectural rule** (user-specified): in RNR mode, gadgets are applied
**only to the free branch**. The fixed branch is always `unsafe(subgame)`
regardless of the chosen gadget. The target (rational) player's info states
are shared across both branches; the opponent's info states are always
branch-local (even at lock=false). This is equivalent to the current CDRNR
math — where the "fixed part" is precomputed via reach weighting and
injected at terminals — just re-expressed explicitly as two branches.

## 1. Target mapping

How existing algorithms map onto the new axes (after refactor):

| Algorithm            | Gadget         | Response | Solver | Lock  | Notes                                               |
|----------------------|----------------|----------|--------|-------|-----------------------------------------------------|
| Safe re-solving      | `Resolving`    | Nash     | CFR    | n/a   | Burch, Johanson, Bowling 2014                       |
| Max-margin           | `MaxMargin`    | Nash     | CFR    | n/a   | Moravcik et al. 2016 / DeepStack                    |
| Unsafe               | `Unsafe`       | Nash     | CFR    | n/a   |                                                     |
| CDBR                 | any            | RNR      | CFR    | true  | `p=1`; opponent in fixed branch is σ^fix as chance  |
| CDRNR                | any            | RNR      | CFR    | true  | `0<p<1`; same fixed-branch treatment                |
| SES (Liu 2022)       | `MaxMargin`    | RNR      | CFR    | false | `p=α`; opponent free in fixed branch (branch-local) |
| OX-search (Ge 2024)  | `ResolvingByIS`| RNR      | CFR    | false | See caveats in §6                                   |
| Full Gadget (LP)     | `FullPath` / `FullTrunk` | Nash | LP | n/a   | Exact equilibrium                                   |
| Full Gadget (CFR)    | `FullPath` / `FullTrunk` | Nash | CFR | n/a   |                                                     |
| Full Gadget + RNR    | `FullPath` / `FullTrunk` | RNR  | CFR / LP | either | Fixed branch = unsafe over the whole decomp     |
| **New: LP RNR**      | any            | RNR      | LP     | either| Enabled by §4.5 wrapper; exact RNR response         |
| **New: LP SES**      | `MaxMargin`    | RNR      | LP     | false | Exact SES via LP on wrapped game                    |

The SES equivalence is exact once you observe that under RNR on a max-margin
gadget:
- Free side (weight `1-p`): adversary picks IS to minimize resolving value
  (the SES *safety branch*).
- Fixed side (weight `p`): wrapper returns a distribution over IS weighted by
  `p̂(I) / Σ_j p̂(I_j)`, which is exactly the SES *exploit branch* (chance
  picks IS by model reach).
- Both sides share the subgame tree, so the non-resolving player automatically
  plays one strategy there — the SES "shared strategy across branches"
  property falls out for free.
- `lock=false` means the wrapper returns empty on subgame info states, so the
  non-resolving player is a free CFR player in the subgame — matches SES.

OX reproduction is **not perfectly numerical** because OX's top-level chance
picks the branch with a weight that depends on `kβ`, not on a simple
RNR-style `p`. The plan exposes a `ResolvingByIS` primitive that captures the
structural core and documents the residual difference. OX exactness is a
nice-to-have, not a hard requirement for this refactor.

## 2. Files

### New

- `open_spiel/game_transforms/gadget.h` — abstract `Gadget` interface and
  `GadgetContext` struct
- `open_spiel/game_transforms/gadget.cc` — shared helpers + factory
  (`MakeGadgetFromKind`), plus the six concrete `*Strategy` classes (wrap
  existing `*_gadget.{h,cc}` code rather than duplicating it)
- `open_spiel/game_transforms/resolving_by_is.h`,
  `open_spiel/game_transforms/resolving_by_is.cc` — new primitive gadget
  (chance-picks-IS + adversary enter/out + subgame). Extracted from `ox_gadget.cc`
  as a structural primitive with *no* α/β/branching; branching comes from RNR
- `open_spiel/game_transforms/restricted_response.h`,
  `open_spiel/game_transforms/restricted_response.cc` — **new** game transform
  implementing RNR as a physical game wrapper (§4.5). This decouples the RNR
  response concept from the CFR-specific `RNRSolver`, enabling LP+RNR and any
  other solver+RNR combination
- `open_spiel/game_transforms/restricted_response_test.cc` — verify the
  wrapper reproduces `RNRSolver` output within `1e-6` on Kuhn across
  `p ∈ {0, 0.25, 0.5, 0.75, 1.0}`
- `open_spiel/game_transforms/gadget_test.cc` — combinatorial test covering
  the orthogonal axes on Kuhn

### Modified

- `open_spiel/game_transforms/continual_resolving.{h,cc}`:
  - `GadgetPolicyWrapper`, `MVSOpponentPolicy`, `MVSModelEntryPolicy`: leave
    alone unless they become unused after the refactor. They're used by
    `ContinualResolve` for trunk-phase RNR on the MVS game, which this
    refactor does not touch
  - Rewrite `ResolveSubgames` as a dispatcher over `Gadget` plus an
    `RNRMixtureGame` composition step for `response=kRNR` (see §6)
  - Update `ResolvingConfig` (see §3)
  - `ContinualResolve` only changes the enum names it passes through
- `open_spiel/game_transforms/CMakeLists.txt` — add new files; remove
  `ses_gadget.{cc,test}` and `ox_gadget.{cc,test}` if we delete them
- `open_spiel/game_transforms/continual_resolving_test.cc` — update
  `ResolvingConfig` field names
- `open_spiel/papers_with_code/depth_limited_responses/dlr_main.cc` — update
  enum names and any SES/OX-specific flags
- `open_spiel/papers_with_code/public_state_cfr/my_main.cc` — only if it
  references the old enums

### Deleted (after migration)

- `open_spiel/game_transforms/ses_gadget.{h,cc,test}` — semantics preserved by
  `MaxMargin + RNR + lock=false` with a reach-weighted IS wrapper. The SES
  test cases move into `gadget_test.cc`
- `open_spiel/game_transforms/ox_gadget.{h,cc,test}` — semantics approximated
  by `ResolvingByIS + RNR + lock=false`. Document the residual numerical
  difference in `gadget_test.cc` or skip strict equality; keep test only as a
  smoke test

If the user objects to deletion, keep the `.cc` files as thin wrappers that
delegate to the new unified path, but remove the custom dispatch logic in
`continual_resolving.cc`.

## 3. `ResolvingConfig` (new)

```cpp
enum class GadgetKind {
  kUnsafe,
  kResolving,       // Burch T/F
  kMaxMargin,       // DeepStack IS pick
  kResolvingByIS,   // new: chance-picks-IS + adversary enter/out (OX primitive)
  kFullPath,
  kFullTrunk,
};

enum class ResponseKind { kNash, kRNR };
enum class SolverKind { kCFR, kLP };

struct ResolvingConfig {
  GadgetKind gadget = GadgetKind::kResolving;
  ResponseKind response = ResponseKind::kNash;
  SolverKind solver = SolverKind::kCFR;

  const Policy* opponent_model = nullptr;  // required for response=kRNR
  double p = 0.5;                          // RNR restriction
  int target_player = 0;                   // RNR: whose strategy is computed
  int cfr_iterations = 500;

  // RNR only: inside the fixed branch's unsafe(subgame), is the opponent
  // a chance node drawn from σ^fix (true) or a free player with
  // branch-local info states (false)?
  //   true  → CDRNR / CDBR — opponent is physically absent in fixed branch
  //   false → SES / OX-style — opponent plays freely but with a distinct
  //           strategy from the free branch (info states are branch-local)
  bool lock_opponent_in_fixed_branch = true;

  // Full gadget only
  int decomp_depth = -1;
};
```

Remove `GadgetType::{kSES, kOX}` and fields `alpha`, `beta` (they are
subsumed by `p`, the gadget selection, and the lock flag). Remove
`SolverType::kRNR` — it moves to `ResponseKind::kRNR`. There is no
`rnr_use_fast_path` escape hatch; the two-branch game is the single
canonical code path.

Add temporary aliases while migrating callers to minimize breakage; delete
them at the end:
```cpp
using GadgetType = GadgetKind;
using SolverType = SolverKind;  // note: no longer has kRNR
```
Legacy callers that use `SolverType::kRNR` must be manually migrated to
`response = kRNR; solver = kCFR;`.

## 4. The `Gadget` interface

```cpp
// open_spiel/game_transforms/gadget.h
#ifndef OPEN_SPIEL_GAME_TRANSFORMS_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_GADGET_H_

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/game_transforms/full_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms { class TabularBestResponse; }

// Per-group state used to build a gadget.
struct GadgetContext {
  std::shared_ptr<const Game> original_game;
  // Roots for this group (owned by caller, NOT by ctx).
  const std::vector<std::unique_ptr<State>>* roots = nullptr;
  std::string pub_obs;

  // From the trunk decomposition.
  std::array<const std::unordered_map<std::string, double>*, 2> cfvs{nullptr, nullptr};
  std::array<const std::unordered_map<std::string, double>*, 2> reach_probs{nullptr, nullptr};
  const std::unordered_map<std::string, double>* chance_reach = nullptr;

  // Which player we're extracting the strategy for in this build.
  Player resolving_player = 0;

  // Optional. Present only when the caller says so.
  const Policy* opponent_model = nullptr;
  const algorithms::TabularBestResponse* br_non_res = nullptr;

  // For MaxMargin SES-reproduction mode.
  bool reach_weighted_is_pick = false;
};

class Gadget {
 public:
  virtual ~Gadget() = default;

  // Build the gadget game. Return nullptr to skip this group.
  virtual std::shared_ptr<const Game> Build(const GadgetContext& ctx) const = 0;

  // Prefix used to key subgame info states inside the gadget game.
  virtual std::string SubgamePrefix() const = 0;

  // True if the non-resolving player's strategy inside the gadget is
  // distorted (because they also use gadget-specific actions). When true,
  // ResolveSubgames must do a second CFR pass with the roles swapped to
  // extract the opponent's undistorted subgame strategy.
  virtual bool OpponentStrategyIsDistorted() const { return true; }

  virtual const char* Name() const = 0;
};

std::unique_ptr<Gadget> MakeUnsafeGadget();
std::unique_ptr<Gadget> MakeResolvingGadget();
std::unique_ptr<Gadget> MakeMaxMarginGadget();
std::unique_ptr<Gadget> MakeResolvingByISGadget();
std::unique_ptr<Gadget> MakeFullGadget(
    FullGadgetGame::Mode mode,
    std::shared_ptr<const Policy> trunk_policy,
    std::unordered_map<std::string, std::vector<std::string>>
        all_boundary_states_by_group);

// Factory from config (selects correct gadget and passes required extras).
std::unique_ptr<Gadget> MakeGadgetFromConfig(
    const ResolvingConfig& config,
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy);

}  // namespace open_spiel
#endif
```

Implementation notes. Each concrete strategy is a thin wrapper around the
existing `*_gadget.cc` code; the `Gadget` interface exists only so the
dispatcher can treat them uniformly. None of them know about RNR or the
fixed branch — that is composed externally by `RNRMixtureGame` (§4.5).

- `UnsafeGadgetStrategy`: `Build` returns `CreateUnsafeSubgame(...)` using
  the joint reach per root (current `ComputeJointReach` helper).
  `OpponentStrategyIsDistorted()` returns `false`.
  `SubgamePrefix()` returns `"unsafe:subgame:"`.

- `ResolvingGadgetStrategy`: `Build` returns `CreateGadgetGame(...)` using
  `BuildSubgameRoots(roots, non_res=1-resolving, reach_res)`.
  `SubgamePrefix()` = `"gadget_F:subgame:"`.

- `MaxMarginGadgetStrategy`: `Build` returns
  `CreateMaxMarginGadgetGame(...)`. `SubgamePrefix()` = `"mm_F:subgame:"`.

- `ResolvingByISStrategy`: new primitive — see §5.

- `FullGadgetStrategy`: `Build` returns
  `CreateFullGadgetGame(game, trunk_policy, resolving_player, pub_obs,
   all_boundary_states_by_group, mode, /*portfolios_p0*/{}, /*portfolios_p1*/{},
   /*enumerate_boundary_portfolios=*/true)`.
  `SubgamePrefix()` = `"full_F:subgame:"`. Extraction strips the optional
  `"P<id>:"` tag (as in the current code).

## 4.5. `RNRMixtureGame` game transform (new)

**Why:** RNR is currently implemented only in `algorithms/rnr.{h,cc}` as a
CFR variant. `RNRSolver::ComputeCounterFactualRegret` threads a
`fixed_opponent_reach` value through the recursion and applies the
`(1-p)*free + p*fixed` weighting only at terminals for the target player.
This ties RNR to CFR. To enable LP+RNR (and, in general, *any*-solver+RNR),
we compose the fixed/free worlds as a physical game with a root chance node.

Crucially — per the user-specified architectural rule — this is also the
place where gadget-vs-unsafe asymmetry lives. The **gadget is applied only
to the free branch.** The fixed branch is always `unsafe(subgame)`. This is
mathematically equivalent to the current CDRNR implementation (where the
fixed part is reach-weighted and injected at terminals only); we just make
it an explicit two-branch game so any solver works.

**Semantics.** Given:
- `free_game`: the gadget game built by the chosen gadget strategy (the
  `*_gadget` game: resolving, max-margin, full, resolving-by-IS, …). Can be
  `unsafe(subgame)` as a no-op gadget.
- `fixed_game`: `CreateUnsafeSubgame(...)` over the same roots. (The fixed
  branch is **always** the unsafe subgame, regardless of `free_game`.)
- `target_player t`.
- `fixed_opponent_policy σ^fix` for the opponent `o = 1 - t` (original game
  info states).
- `p ∈ [0, 1]`.
- `lock_opponent_in_fixed_branch`: controls opponent handling inside the
  fixed branch's unsafe subgame.

`RNRMixtureGame` constructs a new game `G'`:

1. **Root chance** picks `kFree` with probability `(1-p)` or `kFixed` with
   probability `p`.
2. **Free branch** = `free_game`, unmodified. Both players play normally
   there; the opponent is a regular free player (whose strategy CFR/LP will
   solve for).
3. **Fixed branch** = `fixed_game` (unsafe subgame) with opponent handling
   depending on `lock_opponent_in_fixed_branch`:
   - `lock=true`: every opponent decision node in the fixed branch is
     replaced by a chance node with outcomes drawn from `σ^fix(I_o(h))`
     (using the **original** game info state, stripped of the `"unsafe:"`
     prefix). The opponent has no decision nodes in the fixed branch.
   - `lock=false`: the opponent is a normal free player in the fixed
     branch, but its information states are **branch-local** — tagged so
     they are disjoint from both the free branch's opponent info states and
     from anything `σ^fix` keys on. The opponent can therefore play a
     different (best-response) strategy in the fixed branch than in the
     free branch.
4. **Target player's information states are shared across branches for the
   in-subgame nodes**, and branch-local otherwise. Concretely: when the
   target player is asked for an info state at a state whose underlying
   inner-game state is inside a subgame root's subtree, the wrapper returns
   the **original game info state** (no `"gadget_*:"`, no `"unsafe:"`, no
   `"rr_*:"` prefix). Both branches map the same underlying subgame state
   to the same key, so CFR/LP treats them as one info state — which is
   exactly the RNR assumption: the target plays one strategy against both
   opponent worlds. At gadget-specific entry nodes in the free branch (e.g.
   the `"mm_start"` chance node or the `"gadget_choice:..."` T/F opponent
   nodes) the target typically does not act; if it does, the info state is
   the gadget's own info state (branch-local, since those nodes don't exist
   in the fixed branch).
5. **Opponent information states are always branch-local**: tagged
   `"rr_free:<free_is>"` in the free branch (where `<free_is>` is whatever
   the free gadget game reports). In the fixed branch with `lock=false`,
   tagged `"rr_fixed:<orig_is>"` where `<orig_is>` is the original
   (non-prefixed) game info state. With `lock=true`, opponent doesn't act
   in the fixed branch at all.

**Equivalence to current CDRNR.** In CDRNR today, `RNRSolver` is run on the
gadget game, and at target-player terminals the payoff is weighted by
`((1-p)*free_reach + p*fixed_reach)`, where `fixed_reach` threads the
opponent model through the opponent's decision nodes in the gadget game.
Claim: `RNRMixtureGame(free_game=gadget, fixed_game=unsafe, lock=true,
p)` solved by CFR produces the same target policy, because:
- The root chance injects `(1-p, p)` as a reach-probability factor at every
  downstream terminal, matching the `((1-p)*free + p*fixed)` weighting.
- In the fixed branch, the opponent is chance from `σ^fix`, so the
  contribution from a fixed-branch terminal is exactly the expected value
  under `σ^fix` times the target's reach — the same quantity as
  `p*fixed_reach` in the legacy code.
- The fixed branch uses `unsafe(subgame)`, not the gadget game, which
  matches the current behavior: in the current `RNRSolver`, the
  `fixed_reach` is carried through the opponent's actual decisions in the
  gadget, but the gadget's opponent-side distortion (e.g. T/F in the Burch
  gadget, IS pick in max-margin) is exactly what `RNRSolver` skips over by
  its semantics — it uses the *model's* probabilities at opponent gadget
  nodes, which behaviorally reduces the fixed side to "σ^fix acts as a
  chance on the inner subgame states and does not traverse the gadget
  entry". This matches `unsafe(subgame) + σ^fix-as-chance`.
- Target info-state sharing across branches is already implicit in
  `RNRSolver` (the target plays one strategy on the gadget game), so
  sharing in `RNRMixtureGame` preserves it.

This equivalence is asserted as a **hard test gate** in step 2 of §7: CFR
on `RNRMixtureGame(unsafe, unsafe, lock=true, p)` must match
`RNRSolver(unsafe, p)` within `1e-6` on Kuhn (the easiest case, where the
free gadget is itself unsafe). Then step 7 asserts `RNRMixtureGame(gadget,
unsafe, lock=true, p)` matches today's `RNRSolver(gadget, p)` on the full
`continual_resolving_test` suite.

**Interface.**

```cpp
// open_spiel/game_transforms/rnr_mixture.h
namespace open_spiel {

class RNRMixtureGame;  // fwd

// Compose a (gadget-built) free branch with an unsafe-subgame fixed branch
// into one p-weighted game.
//
// Arguments:
//   free_game: any (typically gadget-wrapped) zero-sum game
//   fixed_game: the unsafe subgame over the same roots (built by
//     CreateUnsafeSubgame with joint reach equal to the opponent-model
//     reach, so that the per-root chance distribution matches σ^fix)
//   target_player
//   fixed_opponent_policy: σ^fix, keyed on ORIGINAL game info states
//   p
//   lock_opponent_in_fixed_branch
//   subgame_is_canonicalizer: function(state, player) -> string, returning
//     the ORIGINAL game info state for in-subgame nodes (no gadget/unsafe
//     prefixes) and the empty string for "this is not an in-subgame node,
//     use the wrapped game's own info state". Used to share target info
//     states across branches.
std::shared_ptr<const RNRMixtureGame> CreateRNRMixtureGame(
    std::shared_ptr<const Game> free_game,
    std::shared_ptr<const Game> fixed_game,
    Player target_player,
    std::shared_ptr<const Policy> fixed_opponent_policy,
    double p,
    bool lock_opponent_in_fixed_branch,
    std::function<std::string(const State&, Player)> subgame_is_canonicalizer);

}  // namespace open_spiel
```

**Implementation sketch.** `RNRMixtureGame` is a `Game` subclass (not a
`WrappedGame`, because it owns *two* inner games). Its `State`:
- Holds a `branch` tag (`kRoot`, `kFree`, `kFixed`) and a pointer to the
  currently active inner `State`.
- In `kRoot`: returns `ChanceOutcomes() = [(0, 1-p), (1, p)]`; `DoApplyAction`
  transitions to `kFree` with `free_game.NewInitialState()` or to `kFixed`
  with `fixed_game.NewInitialState()`.
- In `kFree`: delegates everything to the inner free state. `CurrentPlayer`,
  `LegalActions`, `IsTerminal`, `Returns` come straight through.
- In `kFixed`: delegates, **except** opponent nodes when `lock=true`. At
  those nodes, `CurrentPlayer()` returns `kChancePlayerId`; `ChanceOutcomes`
  returns `σ^fix(orig_is(state))`. `DoApplyAction` applies the chosen action
  to the inner state.
- `InformationStateString(player)`:
  - If `player == target`: let `canon = subgame_is_canonicalizer(inner, t)`.
    If non-empty, return `canon` (shared across branches). Else return a
    branch-tagged fallback (`"rr_free_tgt:..."` / `"rr_fixed_tgt:..."` +
    inner IS) — this catches gadget entry nodes that are target-side but
    outside the subgame tree. In practice we don't expect target to act
    there, but handling it keeps the wrapper total.
  - If `player == opponent`: return `"rr_free:" + inner_is` (free branch)
    or `"rr_fixed:" + orig_is` (fixed branch, lock=false). Fixed branch
    lock=true has no opponent nodes.
- `Returns()`: delegate to the inner terminal unchanged. The `(1-p)` / `p`
  factor is injected automatically by the root chance node in any
  solver's reach-probability computation.

**`subgame_is_canonicalizer`.** Provided by the dispatcher: takes the
gadget's `SubgamePrefix()` (for the free branch) and `"unsafe:subgame:"`
(for the fixed branch) and returns the stripped original game info state.
In `RNRMixtureGame::InformationStateString`, it's invoked on the inner
state; the wrapper decides which prefix to strip based on `branch`.

**Max game length.** `1 + max(free_game.MaxGameLength(),
fixed_game.MaxGameLength())`.
`MaxChanceOutcomes()` is `max(2, free_game.MaxChanceOutcomes(),
fixed_game.MaxChanceOutcomes(), max_opponent_legal_actions_in_fixed_game)`.

**Utility range.** Same as the inner games (they share the original
utility range).

**Testing.** New `rnr_mixture_test.cc`:
- **Gate A (wrapper correctness):** For Kuhn, build `unsafe(root)` for
  both `free_game` and `fixed_game`, with `σ^fix = UniformPolicy`,
  `lock=true`, `target ∈ {0, 1}`, `p ∈ {0, 0.25, 0.5, 0.75, 1.0}`. Solve
  with CFR; compare target's average policy against
  `RNRSolver(unsafe, target, σ^fix, p)` CFR output. Tolerance `1e-6`.
  This isolates the two-branch composition from the gadget logic.
- **Gate B (CDRNR equivalence):** For Kuhn, build `resolving(subgame)` as
  `free_game`, `unsafe(subgame)` as `fixed_game`, `lock=true`. Solve
  with CFR; compare against the legacy `RNRSolver(resolving_gadget, p)`
  output (which is exactly today's CDRNR). Tolerance `1e-6`. Repeat with
  `max_margin` as the free gadget.
- **Gate C (LP parity):** Same configurations as Gate A; compare CFR and
  LP target policies within `1e-4`.

## 5. `ResolvingByIS` gadget (new)

A structural primitive extracted from `ox_gadget.cc`. The gadget game:

- **Phase 1 (chance):** pick IS `i` out of `{I_1, ..., I_k}` with
  probabilities `reach_res(I_i) / Σ_j reach_res(I_j)` by default. Resolving
  player sees info state `"ribis_start"` (a single info state, so no info
  leakage from chance). Non-resolving player sees e.g.
  `"ribis_chance:root"`.
- **Phase 2 (non-resolving player):** picks `enter` (1) or `out` (0). The
  info state for the non-resolving player is the chosen `I_i`'s original
  info state; the resolving player sees `"ribis_choice:opponent_choosing"`.
  - `out` → terminal with non-resolving return
    `+CFV(I_i)/reach_res(I_i)` (mirroring the safe-resolving "Terminate"
    payoff). Resolving return gets the opposite shift.
  - `enter` → phase 3.
- **Phase 3 (chance):** pick root `h ∈ I_i` with probability
  `reach_res(h)/reach_res(I_i)`.
- **Phase 4:** subgame plays as normal; returns shifted by
  `±CFV(I_i)/reach_res(I_i)` at terminals; info states prefixed with
  `"ribis_F:subgame:"`.

Compared to OX, this primitive drops the OX-specific `kβ` branch weighting
and the separate exploit branch, keeping only the "chance picks IS by reach,
adversary chooses enter/out per IS" structural primitive. Recombining this
with RNR (via `RNRMixtureGame` composing `resolving_by_is(subgame)` as the
free branch and `unsafe(subgame)` as the fixed branch) reproduces the
OX-search "use reaches but don't lock" flavor; the exact OX weights
require a follow-up addition and are out of scope for this refactor.

`ResolvingByISStrategy` exposes only `Build`, `SubgamePrefix()`, and
`OpponentStrategyIsDistorted() = true` (the enter/out choice is an
adversarial opponent decision that exists only in the gadget game, so
extracting the opponent's undistorted subgame strategy still requires a
second Nash pass — §6).

## 6. The new `ResolveSubgames`

Response and solver are now fully orthogonal, and the gadget is *always*
solved on a single uniform game — either the gadget game directly (Nash
mode) or the `RNRMixtureGame` composition (RNR mode).

```cpp
std::shared_ptr<TabularPolicy> ResolveSubgames(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    const ResolvingConfig& config) {
  auto combined = std::make_shared<TabularPolicy>();
  CopyTrunkInfoStates(trunk_policy, decomp, combined.get());

  auto gadget = MakeGadgetFromConfig(config, decomp, trunk_policy);

  // Which "resolving_player" passes do we need?
  //   Nash, non-distorting gadget  -> one pass (player 0; extract both)
  //   Nash, distorting gadget      -> two passes (0 and 1)
  //   RNR, non-distorting gadget   -> one pass (target)
  //   RNR, distorting gadget       -> two passes: (1) target via RNR
  //                                   mixture game, (2) opponent via a
  //                                   standard Nash pass on the gadget
  //                                   game (to get its undistorted
  //                                   subgame strategy)
  std::vector<Player> passes;
  if (config.response == ResponseKind::kRNR) {
    passes.push_back(config.target_player);
    if (gadget->OpponentStrategyIsDistorted())
      passes.push_back(1 - config.target_player);
  } else {
    passes = gadget->OpponentStrategyIsDistorted()
                 ? std::vector<Player>{0, 1}
                 : std::vector<Player>{0};
  }

  for (Player resolving : passes) {
    const bool rnr_pass =
        config.response == ResponseKind::kRNR &&
        resolving == config.target_player;

    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      GadgetContext ctx =
          BuildGadgetContext(decomp, pub_obs, roots, resolving, config);
      auto free_game = gadget->Build(ctx);
      if (!free_game) continue;

      std::shared_ptr<const Game> game_to_solve;
      std::string is_prefix;  // prefix to strip from target info states

      if (rnr_pass) {
        // Build the fixed branch: unsafe subgame over the same roots, with
        // per-root chance distributed by the opponent model's reach.
        auto fixed_game =
            BuildUnsafeSubgameWithOpponentModelReach(
                ctx, /*opponent=*/1 - config.target_player);

        auto canonicalizer = MakeSubgameISCanonicalizer(
            /*free_prefix=*/gadget->SubgamePrefix(),
            /*fixed_prefix=*/"unsafe:subgame:");

        game_to_solve = CreateRNRMixtureGame(
            free_game, fixed_game,
            /*target_player=*/config.target_player,
            /*fixed_opponent_policy=*/
            std::shared_ptr<const Policy>(config.opponent_model,
                                          [](const Policy*) {}),
            config.p,
            config.lock_opponent_in_fixed_branch,
            canonicalizer);

        // Target info states in the mixture game are the ORIGINAL game
        // info states (no prefix). Opponent extraction isn't needed in
        // this pass — we're only extracting the target.
        is_prefix = "";
      } else {
        game_to_solve = free_game;
        is_prefix = gadget->SubgamePrefix();
      }

      TabularPolicy policy = SolveNashGame(
          *game_to_solve, config.solver, config.cfr_iterations);

      ExtractResolvingStrategyInto(
          policy, is_prefix, resolving, roots, combined.get());
    }
  }
  return combined;
}
```

`SolveNashGame` takes a `SolverKind ∈ {kCFR, kLP}` and runs CFR+ (via
`CFRPlusSolver`) or the existing OR-Tools LP pipeline on the given game.
For the target-player RNR pass, the game is `RNRMixtureGame(free=gadget,
fixed=unsafe)` and SolveNashGame finds a full Nash equilibrium of this
two-branch zero-sum game — the target's equilibrium strategy in that game
is the RNR response.

**Combinations covered:**

| response | solver | gadget-distorting | path                                         |
|----------|--------|-------------------|----------------------------------------------|
| Nash     | CFR    | no                | CFR on gadget game; 1 pass                   |
| Nash     | CFR    | yes               | CFR on gadget game; 2 passes                 |
| Nash     | LP     | no                | LP on gadget game; 1 pass                    |
| Nash     | LP     | yes               | LP on gadget game; 2 passes                  |
| RNR      | CFR    | no                | CFR on `RNRMixtureGame`; 1 pass              |
| RNR      | CFR    | yes               | CFR on `RNRMixtureGame` (target) + CFR on gadget game (opp) |
| RNR      | LP     | no                | LP on `RNRMixtureGame` (**new**); 1 pass     |
| RNR      | LP     | yes               | LP on `RNRMixtureGame` (target) + LP on gadget game (opp) |

The RNR opponent-extraction pass runs on the plain gadget game (not the
mixture), because that pass is a standard Nash computation — the opponent
strategy we want is the one induced by the gadget alone. This matches
current behavior.

`BuildGadgetContext` does the per-group bookkeeping that is currently
duplicated inline:

- Gather reach probs per root into `ctx.reach_probs`.
- If `gadget == MaxMargin` and `response == kRNR` (i.e. SES-style mode):
  precompute `model_info_set_reach` by calling
  `ComputeReachProbabilities(*game, *opponent_model, non_res, roots)` and
  aggregate per info state. The max-margin gadget's `InfoSetList()` is
  already info-state-ordered at construction, so the dispatcher reads off
  `model_info_set_reach` for each IS in order. This governs the per-root
  chance distribution passed into `CreateMaxMarginGadgetGame`.
- If `gadget` is FullPath/FullTrunk: pre-populate
  `all_boundary_states_by_group` into the `FullGadgetStrategy` constructor
  (not ctx — attach at factory time).

`BuildUnsafeSubgameWithOpponentModelReach` builds
`CreateUnsafeSubgame(...)` whose per-root chance probabilities equal
`opponent_model_reach(root) / Σ_r opponent_model_reach(r)`. This ensures
the fixed-branch chance distribution over subgame roots matches the
opponent's model-weighted reach — the same quantity that `RNRSolver`
threads through `fixed_opponent_reach` today. Helper lives in
`subgame_utils.cc`.

`MakeSubgameISCanonicalizer` returns a lambda that, given an inner state
and a player, inspects the state's info state. If it begins with
`free_prefix` or `fixed_prefix`, the prefix is stripped and the remainder
is returned as the canonical (original-game) info state. Otherwise returns
the empty string (signaling "not in the subgame core — use branch-local
fallback").

`ExtractResolvingStrategyInto` encapsulates the `sub_is.compare(0,
prefix.length()...)` loop that is duplicated everywhere, plus the optional
`"P<id>:"` tag handling for Full Gadget. When called with `is_prefix=""`,
it extracts entries whose key is exactly an original game info state —
which is how target entries come out of `RNRMixtureGame`.

## 7. Step-by-step implementation order

Do every step on the `depth_limited_responses` branch. After each step, run
the modules/build commands from `CLAUDE.md` and rerun:

```
matrix_valued_states_test
continual_resolving_test
resolving_gadget_test
max_margin_gadget_test
unsafe_subgame_test
full_gadget_test
ses_gadget_test        # while still present
ox_gadget_test         # while still present
rnr_mixture_test       # once added
gadget_test            # once added
```

If anything breaks, stop and fix before proceeding. **After every step
that passes its tests, append an entry to `PROJECT_PROGRESS.md`** (per
`CLAUDE.md`).

1. **Create empty skeletons** — `gadget.h`, `gadget.cc`, `resolving_by_is.h`,
   `resolving_by_is.cc`, `rnr_mixture.h`, `rnr_mixture.cc`. Add to
   `CMakeLists.txt`. Build; no functional change.

2. **Implement `RNRMixtureGame`** (§4.5). Write `rnr_mixture_test.cc`
   first (test-driven). **Gate A — wrapper correctness:**
   - Free game = `CreateUnsafeSubgame(Kuhn root)`, fixed game =
     `CreateUnsafeSubgame(Kuhn root)`, `σ^fix = UniformPolicy`,
     `lock=true`, target ∈ {0, 1}, `p ∈ {0, 0.25, 0.5, 0.75, 1}`.
   - Solve the mixture with CFR+; compare target's average policy against
     `RNRSolver(unsafe, target, σ^fix, p)` CFR output. Tolerance `1e-6`.
   - `p=0`: target policy equals Nash (free opponent dominates).
   - `p=1`: target policy equals pure BR to `σ^fix`.
   Implement until these pass. **Hard gate** — the two-branch
   composition must be numerically right before anything else.

3. **Gate B — CDRNR equivalence**: extend `rnr_mixture_test.cc`. Set
   free game = `resolving_gadget(Kuhn subgame)`, fixed game =
   `unsafe(Kuhn subgame)`, `lock=true`, all `p` values. Solve CFR and
   compare target policy against the legacy `RNRSolver(resolving_gadget,
   p)` (today's CDRNR fast path), tolerance `1e-6`. Repeat with max-margin
   as the free gadget. **Hard gate** — if this fails, the claim that
   current CDRNR ≡ `RNRMixture(gadget, unsafe, lock=true)` is wrong and
   the plan must be revised before proceeding.

4. **Port Unsafe, Resolving, MaxMargin strategies** to the `Gadget`
   interface — implement `Build`, `SubgamePrefix`,
   `OpponentStrategyIsDistorted`. Do **not** change any existing caller
   yet. Build.

5. **Port FullGadget strategy** — same. Full gadget holds a
   `trunk_policy` and `all_boundary_states_by_group`, so its constructor
   takes them. The factory `MakeFullGadget(mode, ...)` gets them from the
   caller. Build.

6. **Introduce helpers** in `continual_resolving.cc`:
   - `BuildGadgetContext(decomp, pub_obs, roots, resolving, config)`
   - `ExtractResolvingStrategyInto(policy, is_prefix, player, roots, out)`
   - `SolveNashGame(game, SolverKind, cfr_iterations)` — renamed
     `SolveTransformedGame`; only dispatches CFR or LP, no RNR
   - `BuildUnsafeSubgameWithOpponentModelReach(ctx, opponent)`
   - `MakeSubgameISCanonicalizer(free_prefix, fixed_prefix)`
   Do not delete old code paths yet.

7. **Update `ResolvingConfig`**:
   - Add `enum class ResponseKind { kNash, kRNR };` and field
     `ResponseKind response = kNash;`
   - Rename `SolverType` → `SolverKind`; remove its `kRNR` entry
   - Rename `GadgetType` → `GadgetKind`; keep alias temporarily
   - Add `lock_opponent_in_fixed_branch` (default `true`)
   - Remove `alpha`, `beta`
   - Remove `GadgetKind::kSES`, `kOX`
   - **Migrate all callers in this same step**: any place that sets
     `config.solver = SolverType::kRNR` becomes
     `config.response = ResponseKind::kRNR; config.solver = SolverKind::kCFR;`.
     Places using `config.alpha` / `config.beta` / `kSES` / `kOX` must be
     rewritten per §1's mapping table. Build and run tests.

8. **Rewrite `ResolveSubgames`** using the new dispatcher (§6). The RNR
   path builds `RNRMixtureGame(free_gadget, unsafe, …)` and runs
   `SolveNashGame`. **There is no fast path** — the two-branch game is
   the single canonical RNR path for both CFR and LP. At this point all
   existing `continual_resolving_test` cases should still pass (they're
   guaranteed by Gate B). Run the full test suite.

9. **Implement `ResolvingByIS`** (§5) and register it as
   `GadgetKind::kResolvingByIS`. Add smoke test in `gadget_test.cc`:
   `ResolvingByIS + Nash + CFR` on Kuhn produces a non-empty policy with
   sensible exploitability.

10. **Add SES-equivalence test** to `gadget_test.cc`: run
    `ResolveWithSESGadget` (legacy) and `MaxMargin + RNR + CFR +
    lock=false` (unified, with the gadget context's max-margin IS
    distribution set to the model-reach-weighted distribution via
    `BuildGadgetContext`) on Kuhn with `α ∈ {0, 0.5, 1}` (mapped to the
    same `p` on the unified side) and assert policy equality within
    `1e-6`. **Hard gate**.

11. **Add OX smoke test** to `gadget_test.cc`: `ResolvingByIS + RNR +
    CFR + lock=false` on Kuhn with a reasonable `p`. Assert structural
    validity (policy exists on all subgame info states, exploitability
    finite). Numerical equality to legacy OX is **not** required.

12. **LP+RNR test** in `gadget_test.cc`: for a small selection of
    `(gadget, lock)` combinations on Kuhn, run
    `response=RNR, solver=CFR` and `response=RNR, solver=LP` and compare
    target policy within `1e-4` (LP is exact; CFR has iteration error).

13. **Delete `ses_gadget.{h,cc,test}` and `ox_gadget.{h,cc,test}`**;
    remove from `CMakeLists.txt`; remove includes from
    `continual_resolving.cc`. Run tests.

14. **Combinatorial test (`gadget_test.cc`)** — over
    `gadget ∈ {Unsafe, Resolving, MaxMargin, ResolvingByIS, FullPath, FullTrunk}`,
    `response ∈ {Nash, RNR}`,
    `solver ∈ {CFR, LP}`,
    `lock ∈ {true, false}` (RNR only — collapsed for Nash), on Kuhn:
    - Assert the returned policy is non-empty on every subgame info state
    - Assert game value is within numerical bounds
    - Skip `(FullPath/FullTrunk, LP)` if LP scaling makes it infeasible
      on Kuhn; otherwise include

15. **Migrate `dlr_main.cc`, `my_main.cc`, any CI scripts**. Remove the
    `using SolverType = SolverKind;` and `using GadgetType = GadgetKind;`
    aliases. Run `dlr_main` end-to-end on Kuhn/Leduc as a smoke test,
    comparing exploitability to pre-refactor baseline with the same RNG
    seed. Output should match because Gate B asserts CFR-on-mixture ≡
    RNRSolver-on-gadget for the target, and the opponent pass is
    unchanged.

16. **Update `PROJECT_PROGRESS.md`** with a short entry — the refactor,
    what moved where, which files were deleted, which tests validate the
    orthogonal combinations.

## 8. Risks and open questions

- **`RNRMixtureGame` numerical equivalence to `RNRSolver`** (Gate A,
  step 2): the two-branch wrapper must match `RNRSolver` on the
  unsafe/unsafe case. If it fails, the info-state sharing, the
  opponent-as-chance handling, or the root-chance weighting is wrong.
  Most likely culprit: the target-player info state is not actually
  shared across branches (e.g. the canonicalizer leaks a branch tag).
- **CDRNR equivalence** (Gate B, step 3): `RNRMixture(gadget, unsafe,
  lock=true)` must match `RNRSolver(gadget)`. This is the architectural
  claim of §0 — that today's "fixed part reach-weighted into terminals"
  ≡ "fixed branch = unsafe with opponent-as-chance". If it fails on
  max-margin but passes on resolving, the gadget's opponent-side
  distortion at entry (T/F vs. IS pick) interacts with the fixed branch
  differently than assumed; the plan needs revisiting before step 8.
- **SES numerical equivalence** (step 10): the max-margin gadget's per-IS
  chance distribution must be set to the model-reach-weighted
  distribution via `BuildGadgetContext`. Confirm the existing
  `CreateMaxMarginGadgetGame` entry point accepts a custom IS-chance
  distribution; if not, a small extension to `max_margin_gadget.cc` is
  needed first (plumb the distribution through the constructor).
- **OX residual**: OX's specific branch weighting (`kβ/(kβ+1)`) is not
  reproducible by an RNR `p`; we deliberately accept this. Smoke-test
  only; exact numerical reproduction is future work.
- **LP scalability**: LP on the gadget game is only viable for small
  games. The combinatorial test (step 14) may need to skip LP entries
  for Full Gadget on non-trivial games.
- **LP+RNR cost**: `RNRMixtureGame` roughly doubles the sequence space
  (two branches with shared target info states). LP on the mixture may
  be 2–4× slower than LP on the gadget alone. Document in the header.
- **Target canonicalizer correctness**: `MakeSubgameISCanonicalizer`
  must strip both `free_prefix` and `fixed_prefix` to the *same*
  canonical original-game info state. Bug here would silently split the
  target's info sets across branches and quietly destroy the RNR
  guarantee. Assert in a unit test: for a sample of in-subgame states,
  the free-branch and fixed-branch target info states canonicalize to
  equal strings.
- **Fixed-branch unsafe reach distribution**: per-root chance in the
  fixed branch must equal
  `opponent_model_reach(r) / Σ_r opponent_model_reach(r)`. This matches
  the `fixed_reach` quantity that `RNRSolver` threads today; mis-scaling
  it breaks Gate B. `BuildUnsafeSubgameWithOpponentModelReach` is the
  single place this is computed.
- **Per-player passes with `OpponentStrategyIsDistorted=true`**: current
  code does two passes for RNR-with-gadget (target via RNR, opponent via
  CFR). The plan preserves this — the opponent pass runs on the plain
  gadget game, not the mixture. Verify extraction still picks up the
  opponent strategy from the right prefix.
- **Info-state collision with `"rr_*:"` prefixes**: if an inner game
  ever produces info states starting with `"rr_free:"` or `"rr_fixed:"`,
  the opponent tagging would collide. Mitigation: choose prefixes
  unlikely to collide (e.g. `"\x1frrf:"` / `"\x1frrx:"`) or assert no
  collision during `RNRMixtureGame` construction.

## 9. Acceptance criteria

- All pre-existing tests pass (matrix_valued_states, continual_resolving,
  resolving_gadget, max_margin_gadget, unsafe_subgame, full_gadget).
- `rnr_mixture_test.cc` Gate A passes: `RNRMixture(unsafe, unsafe,
  lock=true)` CFR matches `RNRSolver(unsafe)` within `1e-6` on Kuhn
  across `p ∈ {0, 0.25, 0.5, 0.75, 1}` for both target players.
- `rnr_mixture_test.cc` Gate B passes: `RNRMixture(gadget, unsafe,
  lock=true)` CFR matches `RNRSolver(gadget)` within `1e-6` on Kuhn for
  resolving and max-margin gadgets, across `p ∈ {0, 0.25, 0.5, 0.75, 1}`.
- `rnr_mixture_test.cc` Gate C passes: CFR and LP on the mixture game
  agree within `1e-4` on Kuhn.
- New `gadget_test.cc` passes on all orthogonal combinations
  (`gadget × response × solver × lock`).
- SES equivalence test passes within `1e-6` on Kuhn for
  `α ∈ {0, 0.5, 1}`.
- `continual_resolving_test` passes unchanged — no numerical tolerance
  relaxation. This is enforced by Gate B.
- `dlr_main` runs end-to-end with no behavior change on Kuhn/Leduc
  (compare exploitability to pre-refactor baseline, same RNG seed;
  output should be identical modulo float noise because Gate B asserts
  CFR-on-mixture ≡ RNRSolver-on-gadget for the target).
- `ses_gadget.*` and `ox_gadget.*` files are deleted.
- `PROJECT_PROGRESS.md` has a new entry describing the refactor.

## 10. Out of scope

- Exact OX numerical reproduction (documented residual; future work).
- Changing the full gadget's lazy portfolio enumeration — the refactor
  exposes it through the `Gadget` interface unchanged.

## 11. Save-progress discipline

Per `CLAUDE.md`: **after every step above that passes its tests, append a
note to `PROJECT_PROGRESS.md`** describing what was done and the current
state. Do not batch progress notes to the end — conversations can disconnect.
