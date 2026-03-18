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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_OX_GADGET_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_OX_GADGET_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// OX-Search Gadget Game (Ge et al., ICML 2024)
// "Safe and Robust Subgame Exploitation in Imperfect Information Games"
//
// Two-branch gadget for adaptation-safe opponent exploitation:
//   S1 (exploit, prob 1/(kβ+1)): Chance picks IS by model reach → subgame
//   S2 (safety, prob kβ/(kβ+1)): Chance picks IS uniformly → enter/out option → subgame
//
// Resolving player cannot distinguish branches.
// Utility shifted by CBV (Counterfactual Best Response Value).
//
// Game structure:
//   Phase 0 (kInitialChance): Top-level chance node
//     - Action 0 → safety branch (prob kβ/(kβ+1))
//     - Action 1 → exploit branch (prob 1/(kβ+1))
//
//   EXPLOIT BRANCH (S1):
//   Phase 1 (kExploitChanceInfoSet): Chance picks info set I by model reach p̂(I)
//
//   SAFETY BRANCH (S2):
//   Phase 2 (kSafeChanceInfoSet): Chance picks IS uniformly (1/k)
//   Phase 3 (kOptionChoice): Non-resolving player: enter (1) or out (0)
//     - out → terminal with payoff {0, 0}
//     - enter → continue to kChanceWithinInfoSet
//
//   BOTH BRANCHES:
//   Phase 4 (kChanceWithinInfoSet): Chance picks state h ∈ I
//     - Prob(h) = π_{-nonres}(h) / W(I)
//
//   Phase 5 (kSubgame): Normal subgame play with shifted utilities:
//     - Non-resolving return -= CBV(I)/W(I)
//     - Resolving return    += CBV(I)/W(I)
//     - Info states prefixed with "ox_F:subgame:" (SHARED across branches)
//
// SubgameRoots for OX:
//   - info_state_string = NON-resolving player's info state (for grouping)
//   - reach_prob = resolving player's reach (π_{-nonres} = π_res * π_c)
//   - counterfactual values: CBV for NON-resolving player (NOT blueprint CFV!)

namespace open_spiel {

class OXGadgetState;

class OXGadgetGame : public WrappedGame {
 public:
  static constexpr Action kOutAction = 0;    // Safety: take payoff 0
  static constexpr Action kEnterAction = 1;  // Safety: enter subgame

  OXGadgetGame(
      std::shared_ptr<const Game> game,
      std::vector<SubgameRoot> subgame_roots,
      Player resolving_player,
      std::unordered_map<std::string, double> cbv_values,  // CBV, not CFV!
      double beta,  // Lagrange multiplier bound (safety parameter)
      std::unordered_map<std::string, double> model_info_set_reach);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override;

  Player ResolvingPlayer() const { return resolving_player_; }
  Player NonResolvingPlayer() const { return 1 - resolving_player_; }
  int NumSubgameRoots() const { return subgame_roots_.size(); }
  int NumInfoSets() const { return info_set_list_.size(); }
  double Beta() const { return beta_; }

  // Branch probabilities
  double ExploitBranchProb() const;  // 1/(kβ+1)
  double SafetyBranchProb() const;   // kβ/(kβ+1)

  // Get the value shift for a given info set: CBV(I) / W(I)
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
  friend class OXGadgetState;

  void ComputeInfoSetData();

  std::vector<SubgameRoot> subgame_roots_;
  Player resolving_player_;
  std::unordered_map<std::string, double> cbv_values_;  // CBV, not CFV
  double beta_;

  // Ordered list of distinct info state strings (determines action mapping)
  std::vector<std::string> info_set_list_;

  // Map from info state string to index in info_set_list_
  std::unordered_map<std::string, int> info_set_index_;

  // Map from info set index to list of root indices in subgame_roots_
  std::vector<std::vector<int>> info_set_roots_;

  // W(I) = sum of reach probs for roots in info set I (resolving player reach)
  std::unordered_map<std::string, double> info_set_reach_sums_;

  // Value shift = CBV(I) / W(I) for each info set
  std::unordered_map<std::string, double> value_shifts_;

  // model_info_set_reach[info_set] = p̂(I), normalized to sum to 1
  std::unordered_map<std::string, double> model_info_set_reach_;

  // Pre-computed exploit chance outcomes: (info_set_idx, normalized_prob)
  std::vector<std::pair<Action, double>> exploit_chance_outcomes_;

  // Maximum absolute value shift (for utility bounds)
  double max_abs_shift_;
};

class OXGadgetState : public WrappedState {
 public:
  enum class Phase {
    kInitialChance,         // Top chance: exploit vs safety
    kExploitChanceInfoSet,  // [S1] Chance picks IS by model reach
    kSafeChanceInfoSet,     // [S2] Chance picks IS uniformly
    kOptionChoice,          // [S2 only] Non-resolving: enter/out
    kChanceWithinInfoSet,   // Chance picks state h ∈ I
    kSubgame,               // Normal subgame play
    kTerminal               // After "out" or subgame terminal
  };

  explicit OXGadgetState(std::shared_ptr<const Game> game);
  OXGadgetState(const OXGadgetState& other);

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
  bool ChoseOut() const { return chose_out_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const OXGadgetGame* GetOXGame() const;

  Phase phase_;
  int chosen_branch_;          // 0 = safety, 1 = exploit
  int chosen_info_set_idx_;    // Which info set was chosen
  int selected_root_idx_;      // Which root was chosen by chance
  std::string chosen_info_state_;  // Info state string of chosen info set
  bool chose_out_;             // Whether non-resolving player chose "out"
};

// Factory function
std::shared_ptr<const OXGadgetGame> CreateOXGadgetGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> cbv_values,
    double beta,
    std::unordered_map<std::string, double> model_info_set_reach);

// Re-solve all subgames using the OX gadget.
// beta: safety parameter (higher = safer, lower = more exploitation)
// opponent_model: used to compute model reach p̂(I) AND to compute CBV
// trunk_policy: the trunk policy (needed for TabularBestResponse to compute CBV)
std::shared_ptr<TabularPolicy> ResolveWithOXGadget(
    const SubgameDecomposition& decomp,
    const TabularPolicy& trunk_policy,
    double beta,
    const Policy& opponent_model,
    int cfr_iterations = 500);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_OX_GADGET_H_
