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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_RESOLVING_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_RESOLVING_GADGET_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// This transformation implements the "re-solving gadget" from the paper:
// "Solving Imperfect Information Games Using Decomposition"
// (Burch, Johanson, Bowling 2014, arXiv:1303.4441)
//
// The gadget allows safe subgame re-solving while maintaining exploitability
// guarantees. At each root state of the subgame, the "resolving player"
// (opponent whose counterfactual values we want to preserve) gets a choice:
//   - T (terminate): receive the original counterfactual value
//   - F (follow): continue playing in the actual subgame
//
// This guarantees that the re-solved strategy is no more exploitable than
// the original strategy.
//
// Key inputs:
// - subgame_roots: States at the root of the subgame (e.g., start of round 2)
// - resolving_player: The player whose CF values we preserve (opponent)
// - counterfactual_values: Map from info set strings to CF values
// - reach_probabilities: Map from state to reach prob of non-resolving player
//
// Construction (from paper Figure 2):
// 1. Initial chance node distributes to root states with prob π_{-res}(r)/k
// 2. At each root r̃, resolving_player chooses F or T
// 3. T -> terminal with u(T) = k * v^R(I(r)) / Σ_{h∈I(r)} π_{-res}(h)
// 4. F -> continue to subgame, utilities scaled by k
//
// Usage:
//   // After solving trunk, extract counterfactual values and reach probs
//   auto gadget = std::make_shared<GadgetGame>(
//       subgame_roots,           // States at subgame entry
//       resolving_player,        // Player 0 or 1
//       counterfactual_values,   // From trunk solution
//       reach_probabilities);    // From trunk solution
//   // Solve the gadget game to get a safe strategy for the other player

namespace open_spiel {

class GadgetState;

// Information about a root state in the subgame
struct SubgameRoot {
  std::unique_ptr<State> state;
  std::string info_state_string;  // Info state of resolving player
  double reach_prob;              // Reach prob of non-resolving player (π_{-res})
};

class GadgetGame : public WrappedGame {
 public:
  // Actions in the gadget game
  static constexpr Action kTerminateAction = 0;  // T: take original CF value
  static constexpr Action kFollowAction = 1;     // F: continue to subgame

  // Construct a gadget game for safe re-solving.
  //
  // Parameters:
  //   game: The original game (for type info and state cloning)
  //   subgame_roots: Vector of root states with their info sets and reach probs
  //   resolving_player: Player whose CF values we preserve (0 or 1)
  //   counterfactual_values: Map from resolving player's info set string
  //                          to their counterfactual best response value
  GadgetGame(std::shared_ptr<const Game> game,
             std::vector<SubgameRoot> subgame_roots,
             Player resolving_player,
             std::unordered_map<std::string, double> counterfactual_values);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;

  // Access to configuration
  Player ResolvingPlayer() const { return resolving_player_; }
  Player NonResolvingPlayer() const { return 1 - resolving_player_; }
  int NumSubgameRoots() const { return subgame_roots_.size(); }

  // Get the normalization constant k
  double NormalizationConstant() const { return k_; }

  // Get the T-action payoff for a given info set
  double GetTerminatePayoff(const std::string& info_state) const;

  // Get the reach probability for a root state (index into subgame_roots_)
  double GetRootReachProbability(int root_idx) const;

  // Get the info state string for a root state
  const std::string& GetRootInfoState(int root_idx) const;

  // Clone the underlying state at a root
  std::unique_ptr<State> CloneRootState(int root_idx) const;

  // Get minimum and maximum utilities (scaled by k)
  double MinUtility() const override;
  double MaxUtility() const override;

 private:
  friend class GadgetState;

  // Compute the T-action payoff for an info set
  void ComputeTerminatePayoffs();

  std::vector<SubgameRoot> subgame_roots_;
  Player resolving_player_;
  std::unordered_map<std::string, double> counterfactual_values_;

  // Normalization constant k = Σ_r π_{-res}(r)
  double k_;

  // Precomputed T-action payoffs: info_state -> payoff
  // T payoff = k * v^R(I) / Σ_{h∈I} π_{-res}(h)
  std::unordered_map<std::string, double> terminate_payoffs_;

  // Sum of reach probs per info state: info_state -> Σ_{h∈I} π_{-res}(h)
  std::unordered_map<std::string, double> info_state_reach_sums_;
};

class GadgetState : public WrappedState {
 public:
  // Phase tracking for the gadget state machine
  enum class Phase {
    kChance,       // Initial chance node selecting root state
    kGadgetChoice, // Resolving player choosing T or F
    kSubgame,      // Playing the actual subgame (after F)
    kTerminal      // Terminal state (after T or subgame ends)
  };

  GadgetState(std::shared_ptr<const Game> game);
  GadgetState(const GadgetState& other);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<Action> LegalActions(Player player) const override;
  std::string ActionToString(Player player, Action action_id) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::string ToString() const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Access to phase for debugging/testing
  Phase GetPhase() const { return phase_; }
  int SelectedRootIndex() const { return selected_root_idx_; }
  bool ChoseTerminate() const { return chose_terminate_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const GadgetGame* GetGadgetGame() const;

  Phase phase_;
  int selected_root_idx_;    // Which root state was selected by chance
  bool chose_terminate_;      // Whether resolving player chose T
  std::string root_info_state_; // Info state string at the selected root
};

// Factory function to create a gadget game
std::shared_ptr<const GadgetGame> CreateGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> counterfactual_values);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_RESOLVING_GADGET_H_
