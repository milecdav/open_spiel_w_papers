// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_RNR_MIXTURE_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_RNR_MIXTURE_H_

// RNRMixtureGame: compose a (gadget-built) free branch with an unsafe-subgame
// fixed branch into one p-weighted game, enabling RNR with any solver.
//
// Semantics:
//   Root chance picks kFree (prob 1-p) or kFixed (prob p).
//   Free branch: free_game, both players play normally.
//   Fixed branch: fixed_game (unsafe subgame) with:
//     lock=true  → opponent nodes become chance from σ^fix
//     lock=false → opponent is free but with branch-local info states
//   Target player info states are shared across branches for in-subgame nodes
//   (via subgame_is_canonicalizer); opponent info states are always branch-local.
//
// This enables RNR with any solver (CFR or LP), because the mixture is encoded
// as an explicit game structure rather than baked into CFR's recursion.
//
// Equivalence claim (verified by Gate A/B tests):
//   CFR on RNRMixtureGame(unsafe, unsafe, lock=true, p) ≡ RNRSolver(unsafe, p)
//   CFR on RNRMixtureGame(gadget, unsafe, lock=true, p) ≡ RNRSolver(gadget, p)
//
// Note: LP on the mixture game is roughly 2-4x slower than LP on the gadget
// alone due to the doubled sequence space (two branches with shared target ISes).

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

// Branch tags used internally by RNRMixtureState.
enum class RNRBranch { kRoot, kFree, kFixed };

class RNRMixtureGame;

// RNRMixtureState: state of the composed mixture game.
class RNRMixtureState : public State {
 public:
  explicit RNRMixtureState(std::shared_ptr<const Game> game);
  RNRMixtureState(const RNRMixtureState& other);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::string ToString() const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  const RNRMixtureGame* GetMixtureGame() const;

  // Which branch are we in?
  RNRBranch branch_;

  // Inner state (null when branch_ == kRoot).
  std::unique_ptr<State> inner_state_;
};

// RNRMixtureGame: the composed game.
// Owns two inner games (free and fixed); is NOT a WrappedGame.
class RNRMixtureGame : public Game {
 public:
  // Branch actions at the root.
  static constexpr Action kFreeBranchAction = 0;
  static constexpr Action kFixedBranchAction = 1;

  RNRMixtureGame(std::shared_ptr<const Game> free_game,
                 std::shared_ptr<const Game> fixed_game,
                 Player target_player,
                 std::shared_ptr<const Policy> fixed_opponent_policy,
                 double p,
                 bool lock_opponent_in_fixed_branch,
                 double fixed_branch_utility_multiplier,
                 std::function<std::string(const State&, Player)>
                     subgame_is_canonicalizer);

  // Game interface.
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0.0; }
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override;
  std::vector<int> InformationStateTensorShape() const override { return {1}; }

  // Accessors used by RNRMixtureState.
  const Game& FreeGame() const { return *free_game_; }
  const Game& FixedGame() const { return *fixed_game_; }
  Player TargetPlayer() const { return target_player_; }
  Player OpponentPlayer() const { return 1 - target_player_; }
  const Policy& FixedOpponentPolicy() const { return *fixed_opponent_policy_; }
  double P() const { return p_; }
  bool LockOpponentInFixedBranch() const { return lock_opponent_in_fixed_branch_; }
  double FixedBranchUtilityMultiplier() const {
    return fixed_branch_utility_multiplier_;
  }

  // Canonicalize an inner-game info state to the original game info state
  // for in-subgame nodes. Returns empty string if not in subgame.
  std::string Canonicalize(const State& inner_state, Player player) const {
    return subgame_is_canonicalizer_(inner_state, player);
  }

 private:
  friend class RNRMixtureState;

  std::shared_ptr<const Game> free_game_;
  std::shared_ptr<const Game> fixed_game_;
  Player target_player_;
  std::shared_ptr<const Policy> fixed_opponent_policy_;
  double p_;
  bool lock_opponent_in_fixed_branch_;
  double fixed_branch_utility_multiplier_;
  std::function<std::string(const State&, Player)> subgame_is_canonicalizer_;
};

// Compose a (gadget-built) free branch with an unsafe-subgame fixed branch
// into one p-weighted game.
//
// Arguments:
//   free_game: any (typically gadget-wrapped) zero-sum game
//   fixed_game: the unsafe subgame over the same roots
//   target_player: whose RNR response we're computing
//   fixed_opponent_policy: σ^fix, keyed on ORIGINAL game info states
//   p: RNR restriction probability (fixed branch weight)
//   lock_opponent_in_fixed_branch:
//     true  → opponent nodes in fixed branch become chance from σ^fix
//     false → opponent is free in fixed branch with branch-local info states
//   subgame_is_canonicalizer: function(inner_state, player) -> string, returning
//     the ORIGINAL game info state for in-subgame nodes (no gadget/unsafe
//     prefixes) and the empty string for "not an in-subgame node".
std::shared_ptr<const RNRMixtureGame> CreateRNRMixtureGame(
    std::shared_ptr<const Game> free_game,
    std::shared_ptr<const Game> fixed_game,
    Player target_player,
    std::shared_ptr<const Policy> fixed_opponent_policy,
    double p,
    bool lock_opponent_in_fixed_branch,
    double fixed_branch_utility_multiplier,
    std::function<std::string(const State&, Player)> subgame_is_canonicalizer);

// Helper: make a canonicalizer that strips free_prefix or fixed_prefix
// from the inner-game's InformationStateString to recover the original.
// Called with the inner state (from the free or fixed game respectively);
// the canonicalizer uses the state's info state to detect and strip the prefix.
//
// Returns "" when the info state does not begin with either prefix (i.e.,
// the node is a gadget-specific entry node, not an in-subgame node).
std::function<std::string(const State&, Player)>
MakeSubgameISCanonicalizer(const std::string& free_prefix,
                           const std::string& fixed_prefix);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_RNR_MIXTURE_H_
