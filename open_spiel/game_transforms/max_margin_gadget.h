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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_MAX_MARGIN_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_MAX_MARGIN_GADGET_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// This transformation implements the "max-margin gadget" for safe subgame
// re-solving, as described in Moravcik et al. 2016 (DeepStack).
//
// Unlike the original resolving gadget (Burch et al. 2014) which gives the
// resolving player a per-state T/F choice, the max-margin gadget has the
// resolving player choose which info set to "challenge". This maximizes the
// minimum margin (improvement over the blueprint CFV) across all info sets.
//
// Game structure:
//   Phase 1 (kInfoSetChoice): Resolving player picks info set I
//     - Actions: one per distinct info set at subgame roots
//     - Resolving player sees "mm_start" (single info set)
//     - Non-resolving player sees "mm_choice:opponent_choosing"
//
//   Phase 2 (kChance): Chance picks state h ∈ I
//     - Prob(h) = π_{-res}(h) / W(I), where W(I) = Σ_{h'∈I} π_{-res}(h')
//
//   Phase 3 (kSubgame): Normal subgame play from h
//     - At terminal: returns shifted by -CFV(I)/W(I) for resolving player
//       (and +CFV(I)/W(I) for non-resolving, preserving zero-sum)
//
// At the blueprint strategy, the expected shifted value at every info set is 0.
// If the non-resolving player improves over the blueprint, margins become
// positive. The resolving player picks the worst info set, so the game value
// represents the minimum margin across all info sets.

namespace open_spiel {

class MaxMarginGadgetState;

class MaxMarginGadgetGame : public WrappedGame {
 public:
  MaxMarginGadgetGame(
      std::shared_ptr<const Game> game,
      std::vector<SubgameRoot> subgame_roots,
      Player resolving_player,
      std::unordered_map<std::string, double> counterfactual_values);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override;

  Player ResolvingPlayer() const { return resolving_player_; }
  Player NonResolvingPlayer() const { return 1 - resolving_player_; }
  int NumSubgameRoots() const { return subgame_roots_.size(); }
  int NumInfoSets() const { return info_set_list_.size(); }

  // Get the value shift for a given info set: CFV(I) / W(I)
  double GetValueShift(const std::string& info_state) const;

  // Get the ordered list of info set strings (action i -> info_set_list_[i])
  const std::vector<std::string>& InfoSetList() const { return info_set_list_; }

  // Get the info set string for action index
  const std::string& InfoSetForAction(int action_idx) const;

  // Get root indices belonging to an info set (by info set index)
  const std::vector<int>& RootsForInfoSet(int info_set_idx) const;

  // Clone a root state
  std::unique_ptr<State> CloneRootState(int root_idx) const;

  // Get reach probability for a root
  double GetRootReachProbability(int root_idx) const;

  // Get reach sum W(I) for an info set
  double GetInfoSetReachSum(const std::string& info_state) const;

  double MinUtility() const override;
  double MaxUtility() const override;

 private:
  friend class MaxMarginGadgetState;

  void ComputeInfoSetData();

  std::vector<SubgameRoot> subgame_roots_;
  Player resolving_player_;
  std::unordered_map<std::string, double> counterfactual_values_;

  // Ordered list of distinct info state strings (determines action mapping)
  std::vector<std::string> info_set_list_;

  // Map from info state string to index in info_set_list_
  std::unordered_map<std::string, int> info_set_index_;

  // Map from info set index to list of root indices in subgame_roots_
  std::vector<std::vector<int>> info_set_roots_;

  // W(I) = sum of reach probs for roots in info set I
  std::unordered_map<std::string, double> info_set_reach_sums_;

  // Value shift = CFV(I) / W(I) for each info set
  std::unordered_map<std::string, double> value_shifts_;

  // Maximum absolute value shift (for utility bounds)
  double max_abs_shift_;
};

class MaxMarginGadgetState : public WrappedState {
 public:
  enum class Phase {
    kInfoSetChoice,  // Resolving player picks info set
    kChance,         // Chance picks state within info set
    kSubgame,        // Normal subgame play
    kTerminal        // Terminal state
  };

  MaxMarginGadgetState(std::shared_ptr<const Game> game);
  MaxMarginGadgetState(const MaxMarginGadgetState& other);

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

  Phase GetPhase() const { return phase_; }
  int ChosenInfoSetIndex() const { return chosen_info_set_idx_; }
  int SelectedRootIndex() const { return selected_root_idx_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const MaxMarginGadgetGame* GetMaxMarginGame() const;

  Phase phase_;
  int chosen_info_set_idx_;       // Which info set was chosen by resolving player
  int selected_root_idx_;         // Which root was chosen by chance
  std::string chosen_info_state_; // Info state string of chosen info set
};

// Factory function
std::shared_ptr<const MaxMarginGadgetGame> CreateMaxMarginGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> counterfactual_values);

// Re-solve all subgames using the max-margin gadget.
// Same interface as ResolveWithGadget but uses the max-margin formulation.
std::shared_ptr<TabularPolicy> ResolveWithMaxMarginGadget(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    int cfr_iterations = 500);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_MAX_MARGIN_GADGET_H_
