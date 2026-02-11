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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_UNSAFE_SUBGAME_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_UNSAFE_SUBGAME_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/spiel.h"

// This transformation wraps a set of subgame roots with a chance node at the
// top, weighted by reach probabilities. Unlike the "gadget" game, there is no
// T/F mechanism to preserve counterfactual values.
//
// This is called "unsafe" subgame re-solving because the resulting strategy
// may be more exploitable than the original trunk strategy. It's implemented
// for comparison purposes to demonstrate the value of safe (gadget) re-solving.
//
// Usage:
//   // Collect subgame roots and compute reach probabilities
//   std::vector<std::unique_ptr<State>> roots = ...;
//   std::vector<double> reach_probs = ...;  // Total reach for each root
//   auto unsafe_game = CreateUnsafeSubgame(game, roots, reach_probs);
//   // Solve with CFR, strategies may be exploitable when combined with trunk

namespace open_spiel {

class UnsafeSubgameState;

class UnsafeSubgameGame : public WrappedGame {
 public:
  UnsafeSubgameGame(std::shared_ptr<const Game> game,
                    std::vector<std::unique_ptr<State>> roots,
                    std::vector<double> reach_probs);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override { return game_->NumDistinctActions(); }
  int MaxGameLength() const override { return 1 + game_->MaxGameLength(); }

  // Accessors
  int NumSubgameRoots() const { return roots_.size(); }
  double NormalizationConstant() const { return k_; }
  double GetRootReachProbability(int idx) const { return reach_probs_[idx]; }
  std::unique_ptr<State> CloneRootState(int idx) const;

 private:
  friend class UnsafeSubgameState;

  std::vector<std::unique_ptr<State>> roots_;
  std::vector<double> reach_probs_;
  double k_;  // Normalization constant = sum of reach_probs
};

class UnsafeSubgameState : public WrappedState {
 public:
  enum class Phase {
    kChance,   // Initial chance node selecting root state
    kSubgame   // Playing the actual subgame
  };

  explicit UnsafeSubgameState(std::shared_ptr<const Game> game);
  UnsafeSubgameState(const UnsafeSubgameState& other);

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

  // Access to phase for debugging/testing
  Phase GetPhase() const { return phase_; }
  int SelectedRootIndex() const { return selected_root_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  const UnsafeSubgameGame* GetUnsafeSubgameGame() const;

  Phase phase_;
  int selected_root_;
};

// Factory function
std::shared_ptr<const UnsafeSubgameGame> CreateUnsafeSubgame(
    std::shared_ptr<const Game> game,
    std::vector<std::unique_ptr<State>> roots,
    std::vector<double> reach_probs);

// Re-solve all subgames using unsafe (no gadget) re-solving.
// Same interface as ResolveWithGadget but uses direct subgame solving
// weighted by total reach (reach_p0 * reach_p1).
std::shared_ptr<TabularPolicy> ResolveWithUnsafeSubgame(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    int cfr_iterations = 500);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_UNSAFE_SUBGAME_H_
