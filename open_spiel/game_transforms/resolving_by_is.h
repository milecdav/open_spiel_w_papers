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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_RESOLVING_BY_IS_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_RESOLVING_BY_IS_H_

// ResolvingByIS gadget — a structural primitive extracted from ox_gadget.cc.
//
// Game structure:
//   Phase 1 (chance): pick IS i with prob reach_res(I_i) / Σ reach_res(I_j)
//     Resolving player sees "ribis_start" (single IS — no leakage from chance)
//     Non-resolving player sees "ribis_chance:root"
//   Phase 2 (non-resolving player): picks enter(1) or out(0)
//     Non-resolving player IS = the chosen I_i's original IS
//     Resolving player sees "ribis_choice:opponent_choosing"
//     out → terminal with shifted payoff (+CFV(I_i)/reach_res(I_i))
//     enter → phase 3
//   Phase 3 (chance): pick root h ∈ I_i with prob reach_res(h)/reach_res(I_i)
//   Phase 4: subgame plays normally; utilities shifted; IS prefixed "ribis_F:subgame:"
//
// This is a skeleton — full implementation will be added in step 9.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

class ResolvingByISState;

class ResolvingByISGame : public WrappedGame {
 public:
  static constexpr Action kOutAction = 0;
  static constexpr Action kEnterAction = 1;

  ResolvingByISGame(std::shared_ptr<const Game> game,
                    std::vector<SubgameRoot> subgame_roots,
                    Player resolving_player,
                    std::unordered_map<std::string, double> cfvs);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override;
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override;
  double MinUtility() const override;
  double MaxUtility() const override;

  Player ResolvingPlayer() const { return resolving_player_; }
  Player NonResolvingPlayer() const { return 1 - resolving_player_; }
  int NumInfoSets() const { return info_set_list_.size(); }
  const std::string& InfoSetForAction(int action_idx) const;
  const std::vector<int>& RootsForInfoSet(int info_set_idx) const;
  std::unique_ptr<State> CloneRootState(int root_idx) const;
  double GetRootReachProbability(int root_idx) const;
  double GetInfoSetReachSum(const std::string& info_state) const;
  double GetValueShift(const std::string& info_state) const;

 private:
  friend class ResolvingByISState;
  void ComputeInfoSetData();

  std::vector<SubgameRoot> subgame_roots_;
  Player resolving_player_;
  std::unordered_map<std::string, double> cfvs_;
  std::vector<std::string> info_set_list_;
  std::unordered_map<std::string, int> info_set_index_;
  std::vector<std::vector<int>> info_set_roots_;
  std::unordered_map<std::string, double> info_set_reach_sums_;
  std::unordered_map<std::string, double> value_shifts_;
  double max_abs_shift_;
};

class ResolvingByISState : public WrappedState {
 public:
  enum class Phase {
    kChanceInfoSet,
    kOptionChoice,
    kChanceWithinInfoSet,
    kSubgame,
    kTerminal
  };

  explicit ResolvingByISState(std::shared_ptr<const Game> game);
  ResolvingByISState(const ResolvingByISState& other);

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

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const ResolvingByISGame* GetRIBISGame() const;

  Phase phase_;
  int chosen_info_set_idx_;
  int selected_root_idx_;
  std::string chosen_info_state_;
  bool chose_out_;
};

std::shared_ptr<const ResolvingByISGame> CreateResolvingByISGame(
    std::shared_ptr<const Game> game,
    std::vector<SubgameRoot> subgame_roots,
    Player resolving_player,
    std::unordered_map<std::string, double> cfvs);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_RESOLVING_BY_IS_H_
