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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_SES_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_SES_GADGET_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// SES Gadget Game for Safe Exploitation Search (SES).
// Based on: "Safe Opponent-Exploitation Subgame Refinement" (Liu et al., 2022)
//
// SES balances safety and opponent exploitation via a two-branch gadget:
//   - Safety branch (S1, weight 1-α): An adversary (non-resolving player)
//     picks the worst info set; then chance picks a state within it.
//   - Exploitation branch (S2, weight α): Chance picks an info set according
//     to estimated opponent reach p̂; then chance picks a state within it.
//
// The resolving player (the one being refined) CANNOT distinguish between S1
// and S2. Their information states are shared across both branches. The
// non-resolving player CAN distinguish at the top level but is forced to play
// the same strategy in the subgame (enforced by shared subgame info states).
//
// Game structure:
//   Phase 0 (kInitialChance): Top-level chance node
//     - Action 0 → safety branch (prob 1-α)
//     - Action 1 → exploitation branch (prob α)
//
//   SAFETY BRANCH (S1):
//   Phase 1 (kSafeInfoSetChoice): Non-resolving player picks info set I
//     - Actions: one per distinct non-resolving info set at subgame roots
//     - Non-resolving player sees "ses_safe:choosing"
//     - Resolving player sees "ses_start"
//
//   EXPLOITATION BRANCH (S2):
//   Phase 2 (kExploitChanceInfoSet): Chance picks info set by p̂(I)
//     - Chance node (no player decision)
//
//   BOTH BRANCHES:
//   Phase 3 (kChanceWithinInfoSet): Chance picks state h ∈ I
//     - Prob(h) = π_{-nonres}(h) / W(I)
//
//   Phase 4 (kSubgame): Normal subgame play from h with shifted utilities:
//     - Non-resolving return -= CFV_nonres(I) / W(I)
//     - Resolving return    += CFV_nonres(I) / W(I)
//     - Info states prefixed with "ses_F:subgame:" (SHARED across branches)
//
// SubgameRoots for SES:
//   - info_state_string = NON-resolving player's info state (for grouping)
//   - reach_prob = resolving player's reach (π_{-nonres} = π_res * π_c)
//   - counterfactual_values = NON-resolving player's CFVs

namespace open_spiel {

class SESGadgetState;

class SESGadgetGame : public WrappedGame {
 public:
  SESGadgetGame(
      std::shared_ptr<const Game> game,
      std::vector<SubgameRoot> subgame_roots,
      Player resolving_player,
      std::unordered_map<std::string, double> counterfactual_values,
      double alpha,
      std::unordered_map<std::string, double> model_info_set_reach);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override;

  Player ResolvingPlayer() const { return resolving_player_; }
  Player NonResolvingPlayer() const { return 1 - resolving_player_; }
  int NumSubgameRoots() const { return subgame_roots_.size(); }
  int NumInfoSets() const { return info_set_list_.size(); }
  double Alpha() const { return alpha_; }

  // Get the value shift for a given info set: CFV_nonres(I) / W(I)
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

  // Get exploit chance outcomes: (info_set_idx, probability) pairs weighted by
  // model_info_set_reach (normalized)
  const std::vector<std::pair<Action, double>>& ExploitChanceOutcomes() const {
    return exploit_chance_outcomes_;
  }

  double MinUtility() const override;
  double MaxUtility() const override;

 private:
  friend class SESGadgetState;

  void ComputeInfoSetData();

  std::vector<SubgameRoot> subgame_roots_;
  Player resolving_player_;
  std::unordered_map<std::string, double> counterfactual_values_;
  double alpha_;

  // Ordered list of distinct info state strings (determines action mapping)
  std::vector<std::string> info_set_list_;

  // Map from info state string to index in info_set_list_
  std::unordered_map<std::string, int> info_set_index_;

  // Map from info set index to list of root indices in subgame_roots_
  std::vector<std::vector<int>> info_set_roots_;

  // W(I) = sum of reach probs for roots in info set I (resolving player reach)
  std::unordered_map<std::string, double> info_set_reach_sums_;

  // Value shift = CFV_nonres(I) / W(I) for each info set
  std::unordered_map<std::string, double> value_shifts_;

  // model_info_set_reach[info_set] = p̂(I), normalized to sum to 1
  std::unordered_map<std::string, double> model_info_set_reach_;

  // Pre-computed exploit chance outcomes: (info_set_idx, normalized_prob)
  std::vector<std::pair<Action, double>> exploit_chance_outcomes_;

  // Maximum absolute value shift (for utility bounds)
  double max_abs_shift_;
};

class SESGadgetState : public WrappedState {
 public:
  enum class Phase {
    kInitialChance,        // Top chance: (1-α) left (safety), α right (exploit)
    kSafeInfoSetChoice,    // [Safety] Non-resolving player picks info set
    kExploitChanceInfoSet, // [Exploit] Chance picks info set by p̂
    kChanceWithinInfoSet,  // Chance picks state within info set
    kSubgame,              // Normal subgame play with shifted utilities
    kTerminal              // Terminal state
  };

  explicit SESGadgetState(std::shared_ptr<const Game> game);
  SESGadgetState(const SESGadgetState& other);

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
  int ChosenBranch() const { return chosen_branch_; }
  int ChosenInfoSetIndex() const { return chosen_info_set_idx_; }
  int SelectedRootIndex() const { return selected_root_idx_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const SESGadgetGame* GetSESGame() const;

  Phase phase_;
  int chosen_branch_;          // 0 = safety, 1 = exploit
  int chosen_info_set_idx_;    // Which info set was chosen
  int selected_root_idx_;      // Which root was chosen by chance
  std::string chosen_info_state_;  // Info state string of chosen info set
};

// Factory function
std::shared_ptr<const SESGadgetGame> CreateSESGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> counterfactual_values,
    double alpha,
    std::unordered_map<std::string, double> model_info_set_reach);

// Re-solve all subgames using the SES gadget.
// alpha: exploitation level [0,1]. 0=pure safety (max-margin), 1=pure exploit.
// opponent_model: used to compute model reach p̂(I)
std::shared_ptr<TabularPolicy> ResolveWithSESGadget(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    double alpha,
    const Policy& opponent_model,
    int cfr_iterations = 500);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_SES_GADGET_H_
