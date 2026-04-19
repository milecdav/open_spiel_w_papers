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

#include "open_spiel/game_transforms/matrix_valued_states.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {

// Game type for the MVS transformation
const GameType kGameType{
    /*short_name=*/"matrix_valued_states",
    /*long_name=*/"Matrix-Valued States Depth-Limited Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/{},
    /*default_loadable=*/false};

GameType ConvertType(const GameType& type) {
  GameType new_type = kGameType;
  new_type.long_name = "Matrix-Valued States " + type.long_name;
  new_type.chance_mode = type.chance_mode;
  new_type.utility = type.utility;
  new_type.reward_model = GameType::RewardModel::kTerminal;
  new_type.max_num_players = type.max_num_players;
  new_type.min_num_players = type.min_num_players;
  new_type.provides_information_state_string =
      type.provides_information_state_string;
  new_type.provides_observation_string = type.provides_observation_string;
  return new_type;
}

}  // namespace

// ============================================================================
// MVSPortfolioSelector implementation
// ============================================================================

void MVSPortfolioSelector::Activate(int num_p0, int num_p1) {
  SPIEL_CHECK_GT(num_p0, 0);
  SPIEL_CHECK_GT(num_p1, 0);
  phase_ = Phase::kP0Select;
  num_p0_ = num_p0;
  num_p1_ = num_p1;
  p0_choice_ = kInvalidAction;
  p1_choice_ = kInvalidAction;
}

Player MVSPortfolioSelector::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kP0Select: return 0;
    case Phase::kP1Select: return 1;
    case Phase::kTerminal: return kTerminalPlayerId;
    case Phase::kIdle:
      SpielFatalError("MVSPortfolioSelector::CurrentPlayer called while idle");
  }
  SpielFatalError("Unknown phase");
}

std::vector<Action> MVSPortfolioSelector::LegalActions() const {
  std::vector<Action> actions;
  switch (phase_) {
    case Phase::kP0Select:
      for (int i = 0; i < num_p0_; ++i) actions.push_back(i);
      return actions;
    case Phase::kP1Select:
      for (int i = 0; i < num_p1_; ++i) actions.push_back(i);
      return actions;
    case Phase::kTerminal:
      return {};
    case Phase::kIdle:
      SpielFatalError("MVSPortfolioSelector::LegalActions called while idle");
  }
  SpielFatalError("Unknown phase");
}

std::string MVSPortfolioSelector::ActionToString(Player player,
                                                  Action action_id) const {
  return absl::StrCat("P", player, "_portfolio_", action_id);
}

void MVSPortfolioSelector::ApplyAction(Action action_id) {
  switch (phase_) {
    case Phase::kP0Select:
      p0_choice_ = action_id;
      phase_ = Phase::kP1Select;
      break;
    case Phase::kP1Select:
      p1_choice_ = action_id;
      phase_ = Phase::kTerminal;
      break;
    case Phase::kTerminal:
      SpielFatalError("Cannot apply action to terminal MVS selector");
    case Phase::kIdle:
      SpielFatalError("Cannot apply action to idle MVS selector");
  }
}

std::string MVSPortfolioSelector::InformationStateSuffix(Player player) const {
  switch (phase_) {
    case Phase::kP0Select:
      return ":MVS_SEL0";
    case Phase::kP1Select:
      if (player == 0) {
        return absl::StrCat(":MVS_SEL1:", p0_choice_);
      }
      return ":MVS_SEL1";
    case Phase::kTerminal: {
      Action own_choice = (player == 0) ? p0_choice_ : p1_choice_;
      return absl::StrCat(":MVS_T:", own_choice);
    }
    case Phase::kIdle:
      return "";
  }
  SpielFatalError("Unknown phase");
}

// ============================================================================
// MVSGame implementation
// ============================================================================

MVSGame::MVSGame(std::shared_ptr<const Game> game,
                 std::vector<std::shared_ptr<Policy>> portfolios_p1,
                 std::vector<std::shared_ptr<Policy>> portfolios_p2,
                 int depth_limit,
                 DepthMode depth_mode)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      portfolios_p1_(std::move(portfolios_p1)),
      portfolios_p2_(std::move(portfolios_p2)),
      depth_limit_(depth_limit),
      depth_mode_(depth_mode) {
  SPIEL_CHECK_GT(portfolios_p1_.size(), 0);
  SPIEL_CHECK_GT(portfolios_p2_.size(), 0);
  SPIEL_CHECK_GE(depth_limit_, 0);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
}

std::unique_ptr<State> MVSGame::NewInitialState() const {
  return std::make_unique<MVSState>(shared_from_this(),
                                     game_->NewInitialState());
}

int MVSGame::NumDistinctActions() const {
  // Need to accommodate both original game actions and portfolio indices
  return std::max({game_->NumDistinctActions(),
                   static_cast<int>(portfolios_p1_.size()),
                   static_cast<int>(portfolios_p2_.size())});
}

int MVSGame::MaxGameLength() const {
  // Original game length (capped at depth limit) + 2 portfolio selections
  return depth_limit_ + 2;
}

const std::vector<double>& MVSGame::GetPayoff(const State& state,
                                               int p1_idx, int p2_idx) const {
  std::string history_key = state.HistoryString();
  auto it = payoff_cache_.find(history_key);
  if (it == payoff_cache_.end()) {
    ComputePayoffMatrix(state);
    it = payoff_cache_.find(history_key);
  }

  int idx = p1_idx * portfolios_p2_.size() + p2_idx;
  SPIEL_CHECK_LT(idx, it->second.size());
  return it->second[idx];
}

void MVSGame::ComputePayoffMatrix(const State& state) const {
  std::string history_key = state.HistoryString();

  std::vector<std::vector<double>> matrix;
  matrix.reserve(portfolios_p1_.size() * portfolios_p2_.size());

  for (size_t i = 0; i < portfolios_p1_.size(); ++i) {
    for (size_t j = 0; j < portfolios_p2_.size(); ++j) {
      std::vector<const Policy*> policies = {portfolios_p1_[i].get(),
                                              portfolios_p2_[j].get()};
      // Use ExpectedReturns with unlimited depth from this state
      auto returns = algorithms::ExpectedReturns(
          state, policies, /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
      matrix.push_back(returns);
    }
  }

  payoff_cache_[history_key] = std::move(matrix);
}

// ============================================================================
// MVSState implementation
// ============================================================================

MVSState::MVSState(std::shared_ptr<const Game> game,
                   std::unique_ptr<State> state)
    : WrappedState(game, std::move(state)),
      current_depth_(0),
      current_round_(0) {}

MVSState::MVSState(const MVSState& other)
    : WrappedState(other),
      current_depth_(other.current_depth_),
      current_round_(other.current_round_),
      mvs_selector_(other.mvs_selector_),
      selector_activated_(other.selector_activated_) {}

const MVSGame* MVSState::GetMVSGame() const {
  return down_cast<const MVSGame*>(game_.get());
}

bool MVSState::AtDepthLimit() const {
  if (state_->IsTerminal() || state_->IsChanceNode()) {
    return false;
  }
  const MVSGame* game = GetMVSGame();
  if (game->GetDepthMode() == MVSGame::DepthMode::kActionBased) {
    return current_depth_ >= game->DepthLimit();
  } else {
    return current_round_ >= game->DepthLimit();
  }
}

MVSState::Phase MVSState::GetPhase() const {
  if (!selector_activated_) {
    if (AtDepthLimit()) return Phase::kPortfolioP1;
    return Phase::kNormal;
  }
  switch (mvs_selector_.GetPhase()) {
    case MVSPortfolioSelector::Phase::kP0Select: return Phase::kPortfolioP1;
    case MVSPortfolioSelector::Phase::kP1Select: return Phase::kPortfolioP2;
    case MVSPortfolioSelector::Phase::kTerminal: return Phase::kMatrixTerminal;
    default: return Phase::kNormal;
  }
}

Player MVSState::CurrentPlayer() const {
  if (selector_activated_) return mvs_selector_.CurrentPlayer();
  if (AtDepthLimit()) return 0;  // P0 selects first
  return state_->CurrentPlayer();
}

std::vector<Action> MVSState::LegalActions() const {
  if (selector_activated_) return mvs_selector_.LegalActions();
  if (AtDepthLimit()) {
    const MVSGame* game = GetMVSGame();
    std::vector<Action> actions;
    for (int i = 0; i < game->NumPortfoliosP1(); ++i) actions.push_back(i);
    return actions;
  }
  return state_->LegalActions();
}

std::vector<Action> MVSState::LegalActions(Player player) const {
  if (player != CurrentPlayer()) return {};
  return LegalActions();
}

std::string MVSState::ActionToString(Player player, Action action_id) const {
  if (selector_activated_ || AtDepthLimit()) {
    return mvs_selector_.ActionToString(player, action_id);
  }
  return state_->ActionToString(player, action_id);
}

bool MVSState::IsTerminal() const {
  if (selector_activated_) return mvs_selector_.IsTerminal();
  if (state_->IsTerminal()) return true;
  return false;
}

std::vector<double> MVSState::Returns() const {
  if (selector_activated_ && mvs_selector_.IsTerminal()) {
    const MVSGame* game = GetMVSGame();
    return game->GetPayoff(*state_, mvs_selector_.P0Choice(),
                           mvs_selector_.P1Choice());
  }
  if (state_->IsTerminal()) return state_->Returns();
  return std::vector<double>(num_players_, 0.0);
}

std::vector<double> MVSState::Rewards() const {
  if (IsTerminal()) return Returns();
  return std::vector<double>(num_players_, 0.0);
}

std::string MVSState::InformationStateString(Player player) const {
  std::string base = state_->InformationStateString(player);
  if (selector_activated_ || AtDepthLimit()) {
    // At depth limit but selector not yet activated = same as kP0Select
    if (!selector_activated_) return absl::StrCat(base, ":MVS_SEL0");
    return absl::StrCat(base, mvs_selector_.InformationStateSuffix(player));
  }
  return base;
}

std::string MVSState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string MVSState::ToString() const {
  std::string result = state_->ToString();
  Phase phase = GetPhase();
  switch (phase) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        absl::StrAppend(&result, "\n[MVS: At depth limit, P0 selecting]");
      }
      break;
    case Phase::kPortfolioP1:
      absl::StrAppend(&result, "\n[MVS: P0 selecting portfolio]");
      break;
    case Phase::kPortfolioP2:
      absl::StrAppend(&result, "\n[MVS: P1 selecting portfolio]");
      break;
    case Phase::kMatrixTerminal:
      absl::StrAppend(&result, "\n[MVS: Terminal, P0=",
                      mvs_selector_.P0Choice(), ", P1=",
                      mvs_selector_.P1Choice(), "]");
      break;
  }
  return result;
}

std::unique_ptr<State> MVSState::Clone() const {
  return std::make_unique<MVSState>(*this);
}

std::vector<std::pair<Action, double>> MVSState::ChanceOutcomes() const {
  if (selector_activated_ || AtDepthLimit()) return {};
  return state_->ChanceOutcomes();
}

void MVSState::DoApplyAction(Action action_id) {
  if (selector_activated_) {
    mvs_selector_.ApplyAction(action_id);
    return;
  }

  if (AtDepthLimit()) {
    // First action at depth limit: activate selector, apply P0's choice
    const MVSGame* game = GetMVSGame();
    mvs_selector_.Activate(game->NumPortfoliosP1(), game->NumPortfoliosP2());
    selector_activated_ = true;
    mvs_selector_.ApplyAction(action_id);
    return;
  }

  // Normal game action
  bool was_chance = state_->IsChanceNode();
  state_->ApplyAction(action_id);

  if (!was_chance) {
    current_depth_++;
  } else if (GetMVSGame()->GetDepthMode() == MVSGame::DepthMode::kRoundBased) {
    current_round_++;
  }

  // Check if we should activate selector after this action
  if (!state_->IsTerminal() && !state_->IsChanceNode() && AtDepthLimit()) {
    const MVSGame* game = GetMVSGame();
    mvs_selector_.Activate(game->NumPortfoliosP1(), game->NumPortfoliosP2());
    selector_activated_ = true;
  }
}

std::shared_ptr<const MVSGame> CreateMVSGame(
    std::shared_ptr<const Game> game,
    std::vector<std::shared_ptr<Policy>> portfolios_p1,
    std::vector<std::shared_ptr<Policy>> portfolios_p2,
    int depth_limit,
    MVSGame::DepthMode depth_mode) {
  return std::make_shared<MVSGame>(std::move(game), std::move(portfolios_p1),
                                   std::move(portfolios_p2), depth_limit,
                                   depth_mode);
}

// ============================================================================
// Pure Strategy Enumeration Implementation
// ============================================================================

SubtreePureStrategy::SubtreePureStrategy(PureStrategyMap strategy, Player player)
    : strategy_(std::move(strategy)), player_(player) {}

ActionsAndProbs SubtreePureStrategy::GetStatePolicy(
    const State& state, Player player) const {
  if (player != player_) {
    SpielFatalError("SubtreePureStrategy queried for wrong player");
  }
  std::string info_state = state.InformationStateString(player);
  auto it = strategy_.find(info_state);
  if (it == strategy_.end()) {
    SpielFatalError("SubtreePureStrategy missing infoset in subtree policy");
  }
  // Return full policy with the chosen action having prob 1.0
  // and all other legal actions having prob 0.0
  ActionsAndProbs policy;
  Action chosen_action = it->second;
  for (Action a : state.LegalActions(player)) {
    policy.push_back({a, (a == chosen_action) ? 1.0 : 0.0});
  }
  return policy;
}

ActionsAndProbs SubtreePureStrategy::GetStatePolicy(
    const std::string& info_state) const {
  auto it = strategy_.find(info_state);
  if (it == strategy_.end()) {
    // Infostate not in our subtree - this can happen if the policy is queried
    // for states outside the subtree where it was defined. We return empty,
    // and the caller should handle this (e.g., use a fallback policy).
    // Note: For ExpectedReturns to work, the state-based GetStatePolicy
    // override handles this by returning uniform for unknown states.
    return {};
  }
  // Return deterministic policy on the chosen action
  return {{it->second, 1.0}};
}

namespace {

// Helper to recursively collect infostates
void CollectInfostatesRecursive(
    const State& state,
    Player player,
    SubtreeInfostates* result) {
  if (state.IsTerminal()) {
    return;
  }

  if (state.IsChanceNode()) {
    // Traverse all chance outcomes
    for (const auto& outcome : state.ChanceOutcomes()) {
      std::unique_ptr<State> child = state.Child(outcome.first);
      CollectInfostatesRecursive(*child, player, result);
    }
  } else if (state.CurrentPlayer() == player) {
    // This is our player's decision node
    std::string info_state = state.InformationStateString(player);
    const std::vector<Action> legal = state.LegalActions();

    // Only add if not already seen (infostates can be reached multiple ways)
    auto it = result->legal_actions.find(info_state);
    if (it == result->legal_actions.end()) {
      result->infostates.push_back(info_state);
      result->legal_actions[info_state] = legal;
    } else {
      // Invariant: identical infoset string must imply identical legal actions.
      // Keep sequence equality (not just set equality) so action indexing remains
      // stable for pure-strategy enumeration.
      const std::vector<Action>& prev = it->second;
      SPIEL_CHECK_EQ(prev.size(), legal.size());
      for (int k = 0; k < static_cast<int>(prev.size()); ++k) {
        SPIEL_CHECK_EQ(prev[k], legal[k]);
      }
    }

    // Traverse all actions
    for (Action action : legal) {
      std::unique_ptr<State> child = state.Child(action);
      CollectInfostatesRecursive(*child, player, result);
    }
  } else {
    // Other player's decision node - traverse all actions
    for (Action action : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(action);
      CollectInfostatesRecursive(*child, player, result);
    }
  }
}

// Helper to recursively enumerate pure strategies
void EnumerateRecursive(
    const SubtreeInfostates& infostates,
    size_t infostate_idx,
    PureStrategyMap* current_strategy,
    std::vector<PureStrategyMap>* result) {
  if (infostate_idx >= infostates.infostates.size()) {
    // We've assigned actions to all infostates - add this strategy
    result->push_back(*current_strategy);
    return;
  }

  const std::string& info_state = infostates.infostates[infostate_idx];
  const std::vector<Action>& actions = infostates.legal_actions.at(info_state);

  // Try each legal action at this infostate
  for (Action action : actions) {
    (*current_strategy)[info_state] = action;
    EnumerateRecursive(infostates, infostate_idx + 1, current_strategy, result);
  }

  // Clean up (optional, since we overwrite anyway)
  current_strategy->erase(info_state);
}

}  // namespace

SubtreeInfostates CollectSubtreeInfostates(const State& state, Player player) {
  SubtreeInfostates result;
  CollectInfostatesRecursive(state, player, &result);
  return result;
}

std::vector<PureStrategyMap> EnumeratePureStrategies(
    const SubtreeInfostates& infostates) {
  std::vector<PureStrategyMap> result;

  if (infostates.infostates.empty()) {
    // No decision points for this player - return single empty strategy
    result.push_back({});
    return result;
  }

  PureStrategyMap current_strategy;
  EnumerateRecursive(infostates, 0, &current_strategy, &result);
  return result;
}

std::vector<std::shared_ptr<Policy>> ConvertToPortfolio(
    const std::vector<PureStrategyMap>& pure_strategies, Player player) {
  std::vector<std::shared_ptr<Policy>> portfolio;
  portfolio.reserve(pure_strategies.size());
  for (const auto& strategy : pure_strategies) {
    portfolio.push_back(std::make_shared<SubtreePureStrategy>(strategy, player));
  }
  return portfolio;
}

std::vector<std::shared_ptr<Policy>> EnumerateSubtreePureStrategies(
    const State& state, Player player) {
  SubtreeInfostates infostates = CollectSubtreeInfostates(state, player);
  std::vector<PureStrategyMap> pure_strategies =
      EnumeratePureStrategies(infostates);
  return ConvertToPortfolio(pure_strategies, player);
}

namespace {

void CollectReachableInfosetsUnderPureStrategy(
    const State& state,
    Player player,
    const PureStrategyMap& strategy,
    std::unordered_set<std::string>* reachable_infosets) {
  if (state.IsTerminal()) return;

  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      CollectReachableInfosetsUnderPureStrategy(
          *child, player, strategy, reachable_infosets);
    }
    return;
  }

  Player cur = state.CurrentPlayer();
  if (cur == player) {
    std::string is = state.InformationStateString(player);
    reachable_infosets->insert(is);
    auto it = strategy.find(is);
    if (it == strategy.end()) {
      SpielFatalError("Pure strategy missing reachable infoset: " + is);
    }
    auto child = state.Clone();
    child->ApplyAction(it->second);
    CollectReachableInfosetsUnderPureStrategy(
        *child, player, strategy, reachable_infosets);
    return;
  }

  for (Action a : state.LegalActions()) {
    auto child = state.Clone();
    child->ApplyAction(a);
    CollectReachableInfosetsUnderPureStrategy(
        *child, player, strategy, reachable_infosets);
  }
}

std::string BuildRealizationSignature(
    const PureStrategyMap& strategy,
    const std::unordered_set<std::string>& reachable_infosets) {
  std::vector<std::string> sorted_infosets(reachable_infosets.begin(),
                                           reachable_infosets.end());
  std::sort(sorted_infosets.begin(), sorted_infosets.end());
  std::string sig;
  for (const auto& is : sorted_infosets) {
    auto it = strategy.find(is);
    SPIEL_CHECK_TRUE(it != strategy.end());
    sig += is;
    sig += "=";
    sig += std::to_string(it->second);
    sig += ";";
  }
  return sig;
}

}  // namespace

std::vector<std::shared_ptr<Policy>> EnumerateSubtreePureStrategiesReduced(
    const State& state, Player player) {
  SubtreeInfostates infostates = CollectSubtreeInfostates(state, player);
  std::vector<PureStrategyMap> pure_strategies =
      EnumeratePureStrategies(infostates);

  std::vector<PureStrategyMap> reduced;
  reduced.reserve(pure_strategies.size());
  std::unordered_set<std::string> seen_signatures;
  for (const auto& strategy : pure_strategies) {
    std::unordered_set<std::string> reachable_infosets;
    CollectReachableInfosetsUnderPureStrategy(
        state, player, strategy, &reachable_infosets);
    std::string signature =
        BuildRealizationSignature(strategy, reachable_infosets);
    if (seen_signatures.insert(signature).second) {
      reduced.push_back(strategy);
    }
  }
  return ConvertToPortfolio(reduced, player);
}

// ============================================================================
// MVSGameWithSubtreePureStrategies Implementation
// ============================================================================

namespace {

GameType ConvertTypeForSubtree(const GameType& type) {
  GameType new_type{
      /*short_name=*/"mvs_subtree_pure",
      /*long_name=*/"MVS with Subtree Pure Strategies " + type.long_name,
      GameType::Dynamics::kSequential,
      GameType::ChanceMode::kSampledStochastic,
      GameType::Information::kImperfectInformation,
      GameType::Utility::kGeneralSum,
      GameType::RewardModel::kTerminal,
      /*max_num_players=*/2,
      /*min_num_players=*/2,
      /*provides_information_state_string=*/true,
      /*provides_information_state_tensor=*/false,
      /*provides_observation_string=*/true,
      /*provides_observation_tensor=*/false,
      /*parameter_specification=*/{},
      /*default_loadable=*/false};
  new_type.chance_mode = type.chance_mode;
  new_type.utility = type.utility;
  return new_type;
}

// Helper to get action probability from ActionsAndProbs
double GetActionProb(const ActionsAndProbs& policy, Action action) {
  for (const auto& [a, p] : policy) {
    if (a == action) return p;
  }
  return 0.0;
}

// Recursively compute all payoffs in a single tree traversal.
// reach_p0[i] = reach probability for policy i of player 0
// reach_p1[j] = reach probability for policy j of player 1
// payoff_matrix[i][j] = accumulated expected returns for policy pair (i,j)
void ComputePayoffMatrixBatchRecursive(
    const State& state,
    const std::vector<std::shared_ptr<Policy>>& p0_portfolio,
    const std::vector<std::shared_ptr<Policy>>& p1_portfolio,
    const std::vector<double>& reach_p0,
    const std::vector<double>& reach_p1,
    double chance_reach,
    std::vector<std::vector<std::vector<double>>>* payoff_matrix) {

  size_t n0 = p0_portfolio.size();
  size_t n1 = p1_portfolio.size();

  if (state.IsTerminal()) {
    auto returns = state.Returns();
    for (size_t i = 0; i < n0; ++i) {
      for (size_t j = 0; j < n1; ++j) {
        double weight = reach_p0[i] * reach_p1[j] * chance_reach;
        for (size_t p = 0; p < returns.size(); ++p) {
          (*payoff_matrix)[i][j][p] += weight * returns[p];
        }
      }
    }
    return;
  }

  if (state.IsChanceNode()) {
    for (const auto& [action, prob] : state.ChanceOutcomes()) {
      auto next_state = state.Clone();
      next_state->ApplyAction(action);
      ComputePayoffMatrixBatchRecursive(
          *next_state, p0_portfolio, p1_portfolio,
          reach_p0, reach_p1, chance_reach * prob,
          payoff_matrix);
    }
    return;
  }

  Player player = state.CurrentPlayer();
  auto legal_actions = state.LegalActions();

  // Precompute action probabilities for all policies
  std::vector<ActionsAndProbs> p0_policies(n0);
  std::vector<ActionsAndProbs> p1_policies(n1);
  if (player == 0) {
    for (size_t i = 0; i < n0; ++i) {
      p0_policies[i] = p0_portfolio[i]->GetStatePolicy(state);
    }
  } else {
    for (size_t j = 0; j < n1; ++j) {
      p1_policies[j] = p1_portfolio[j]->GetStatePolicy(state);
    }
  }

  // For each action, compute new reach probabilities and recurse
  for (Action action : legal_actions) {
    std::vector<double> new_reach_p0 = reach_p0;
    std::vector<double> new_reach_p1 = reach_p1;

    if (player == 0) {
      for (size_t i = 0; i < n0; ++i) {
        new_reach_p0[i] *= GetActionProb(p0_policies[i], action);
      }
    } else {
      for (size_t j = 0; j < n1; ++j) {
        new_reach_p1[j] *= GetActionProb(p1_policies[j], action);
      }
    }

    auto next_state = state.Clone();
    next_state->ApplyAction(action);
    ComputePayoffMatrixBatchRecursive(
        *next_state, p0_portfolio, p1_portfolio,
        new_reach_p0, new_reach_p1, chance_reach,
        payoff_matrix);
  }
}

// Compute the entire payoff matrix for a state in a single tree traversal.
// Returns a flattened map: flat_idx -> returns vector
// where flat_idx = i * 10000 + j for portfolio indices (i, j)
std::unordered_map<int, std::vector<double>> ComputePayoffMatrixBatch(
    const State& state,
    const std::vector<std::shared_ptr<Policy>>& p0_portfolio,
    const std::vector<std::shared_ptr<Policy>>& p1_portfolio) {

  size_t n0 = p0_portfolio.size();
  size_t n1 = p1_portfolio.size();
  int num_players = state.NumPlayers();

  // Initialize payoff matrix to zeros
  std::vector<std::vector<std::vector<double>>> payoff_matrix(n0);
  for (size_t i = 0; i < n0; ++i) {
    payoff_matrix[i].resize(n1);
    for (size_t j = 0; j < n1; ++j) {
      payoff_matrix[i][j].assign(num_players, 0.0);
    }
  }

  // Initial reach probabilities are all 1.0
  std::vector<double> reach_p0(n0, 1.0);
  std::vector<double> reach_p1(n1, 1.0);

  ComputePayoffMatrixBatchRecursive(
      state, p0_portfolio, p1_portfolio,
      reach_p0, reach_p1, 1.0,
      &payoff_matrix);

  // Convert to flattened map format
  std::unordered_map<int, std::vector<double>> result;
  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      int flat_idx = i * 10000 + j;
      result[flat_idx] = std::move(payoff_matrix[i][j]);
    }
  }
  return result;
}

}  // namespace

MVSGameWithSubtreePureStrategies::MVSGameWithSubtreePureStrategies(
    std::shared_ptr<const Game> game,
    int depth_limit,
    DepthMode depth_mode,
    std::shared_ptr<Policy> opponent_model,
    Player opponent_player)
    : WrappedGame(game, ConvertTypeForSubtree(game->GetType()),
                  game->GetParameters()),
      depth_limit_(depth_limit),
      depth_mode_(depth_mode),
      opponent_model_(std::move(opponent_model)),
      opponent_player_(opponent_player) {
  SPIEL_CHECK_GE(depth_limit_, 0);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
}

std::unique_ptr<State> MVSGameWithSubtreePureStrategies::NewInitialState() const {
  return std::make_unique<MVSStateWithSubtreePureStrategies>(
      shared_from_this(), game_->NewInitialState());
}

int MVSGameWithSubtreePureStrategies::NumDistinctActions() const {
  // We don't know the max portfolio size ahead of time, but it's at least
  // as many as the original game's actions. In practice, portfolios can be
  // larger, but actions are just indices so this should work.
  return std::max(game_->NumDistinctActions(), 1000);  // Upper bound estimate
}

int MVSGameWithSubtreePureStrategies::MaxGameLength() const {
  return depth_limit_ + 2;
}

const std::vector<double>& MVSGameWithSubtreePureStrategies::GetPayoff(
    const MVSStateWithSubtreePureStrategies& mvs_state,
    int p1_idx, int p2_idx) const {
  const State& underlying = mvs_state.GetUnderlyingState();
  std::string history_key = underlying.HistoryString();
  int flat_idx = p1_idx * 10000 + p2_idx;  // Simple encoding

  // Check if we have this specific payoff cached
  auto it = payoff_cache_.find(history_key);
  if (it != payoff_cache_.end()) {
    auto it2 = it->second.find(flat_idx);
    if (it2 != it->second.end()) {
      return it2->second;
    }
  }

  // Not cached - compute the ENTIRE payoff matrix in one tree traversal
  // This is much faster than computing each (i,j) pair separately
  const auto& p0_portfolio = mvs_state.GetPortfolioP0();
  const auto& p1_portfolio = mvs_state.GetPortfolioP1();

  auto full_matrix = ComputePayoffMatrixBatch(underlying, p0_portfolio, p1_portfolio);

  // Store all computed payoffs in cache
  payoff_cache_[history_key] = std::move(full_matrix);

  return payoff_cache_[history_key][flat_idx];
}

// ============================================================================
// MVSStateWithSubtreePureStrategies Implementation
// ============================================================================

MVSStateWithSubtreePureStrategies::MVSStateWithSubtreePureStrategies(
    std::shared_ptr<const Game> game,
    std::unique_ptr<State> state)
    : WrappedState(game, std::move(state)),
      current_depth_(0),
      current_round_(0),
      portfolios_computed_(false) {}

MVSStateWithSubtreePureStrategies::MVSStateWithSubtreePureStrategies(
    const MVSStateWithSubtreePureStrategies& other)
    : WrappedState(other),
      current_depth_(other.current_depth_),
      current_round_(other.current_round_),
      mvs_selector_(other.mvs_selector_),
      selector_activated_(other.selector_activated_),
      portfolio_p0_(other.portfolio_p0_),
      portfolio_p1_(other.portfolio_p1_),
      portfolios_computed_(other.portfolios_computed_) {}

const MVSGameWithSubtreePureStrategies*
MVSStateWithSubtreePureStrategies::GetMVSGame() const {
  return down_cast<const MVSGameWithSubtreePureStrategies*>(game_.get());
}

bool MVSStateWithSubtreePureStrategies::AtDepthLimit() const {
  if (state_->IsTerminal() || state_->IsChanceNode()) {
    return false;
  }

  const auto* game = GetMVSGame();
  if (game->GetDepthMode() == MVSGame::DepthMode::kActionBased) {
    return current_depth_ >= game->DepthLimit();
  } else {
    return current_round_ >= game->DepthLimit();
  }
}

void MVSStateWithSubtreePureStrategies::EnsurePortfoliosComputed() const {
  if (portfolios_computed_) return;

  // Enumerate pure strategies for both players from the current state
  portfolio_p0_ = EnumerateSubtreePureStrategies(*state_, 0);
  portfolio_p1_ = EnumerateSubtreePureStrategies(*state_, 1);

  // Add opponent model as extra portfolio entry if set
  const auto* game = GetMVSGame();
  if (game->OpponentModel()) {
    if (game->OpponentPlayer() == 0) {
      portfolio_p0_.push_back(game->OpponentModel());
    } else {
      portfolio_p1_.push_back(game->OpponentModel());
    }
  }

  portfolios_computed_ = true;
}

const std::vector<std::shared_ptr<Policy>>&
MVSStateWithSubtreePureStrategies::GetPortfolioP0() const {
  EnsurePortfoliosComputed();
  return portfolio_p0_;
}

const std::vector<std::shared_ptr<Policy>>&
MVSStateWithSubtreePureStrategies::GetPortfolioP1() const {
  EnsurePortfoliosComputed();
  return portfolio_p1_;
}

MVSStateWithSubtreePureStrategies::Phase
MVSStateWithSubtreePureStrategies::GetPhase() const {
  if (!selector_activated_) {
    if (AtDepthLimit()) return Phase::kPortfolioP1;
    return Phase::kNormal;
  }
  switch (mvs_selector_.GetPhase()) {
    case MVSPortfolioSelector::Phase::kP0Select: return Phase::kPortfolioP1;
    case MVSPortfolioSelector::Phase::kP1Select: return Phase::kPortfolioP2;
    case MVSPortfolioSelector::Phase::kTerminal: return Phase::kMatrixTerminal;
    default: return Phase::kNormal;
  }
}

Player MVSStateWithSubtreePureStrategies::CurrentPlayer() const {
  if (selector_activated_) return mvs_selector_.CurrentPlayer();
  if (AtDepthLimit()) return 0;
  return state_->CurrentPlayer();
}

std::vector<Action> MVSStateWithSubtreePureStrategies::LegalActions() const {
  if (selector_activated_) return mvs_selector_.LegalActions();
  if (AtDepthLimit()) {
    EnsurePortfoliosComputed();
    std::vector<Action> actions;
    for (size_t i = 0; i < portfolio_p0_.size(); ++i) actions.push_back(i);
    return actions;
  }
  return state_->LegalActions();
}

std::vector<Action> MVSStateWithSubtreePureStrategies::LegalActions(
    Player player) const {
  if (player != CurrentPlayer()) return {};
  return LegalActions();
}

std::string MVSStateWithSubtreePureStrategies::ActionToString(
    Player player, Action action_id) const {
  if (selector_activated_ || AtDepthLimit()) {
    return absl::StrCat("P", player, "_pure_", action_id);
  }
  return state_->ActionToString(player, action_id);
}

bool MVSStateWithSubtreePureStrategies::IsTerminal() const {
  if (selector_activated_) return mvs_selector_.IsTerminal();
  if (state_->IsTerminal()) return true;
  return false;
}

std::vector<double> MVSStateWithSubtreePureStrategies::Returns() const {
  if (selector_activated_ && mvs_selector_.IsTerminal()) {
    const auto* game = GetMVSGame();
    return game->GetPayoff(*this, mvs_selector_.P0Choice(),
                           mvs_selector_.P1Choice());
  }
  if (state_->IsTerminal()) return state_->Returns();
  return std::vector<double>(num_players_, 0.0);
}

std::vector<double> MVSStateWithSubtreePureStrategies::Rewards() const {
  if (IsTerminal()) return Returns();
  return std::vector<double>(num_players_, 0.0);
}

std::string MVSStateWithSubtreePureStrategies::InformationStateString(
    Player player) const {
  std::string base = state_->InformationStateString(player);
  if (selector_activated_ || AtDepthLimit()) {
    if (!selector_activated_) return absl::StrCat(base, ":MVS_SEL0");
    return absl::StrCat(base, mvs_selector_.InformationStateSuffix(player));
  }
  return base;
}

std::string MVSStateWithSubtreePureStrategies::ObservationString(
    Player player) const {
  return InformationStateString(player);
}

std::string MVSStateWithSubtreePureStrategies::ToString() const {
  std::string result = state_->ToString();
  Phase phase = GetPhase();
  switch (phase) {
    case Phase::kNormal:
      if (AtDepthLimit()) {
        EnsurePortfoliosComputed();
        absl::StrAppend(&result, "\n[MVSP: Depth limit, P0 has ",
                        portfolio_p0_.size(), " strategies, P1 has ",
                        portfolio_p1_.size(), " strategies]");
      }
      break;
    case Phase::kPortfolioP1:
      absl::StrAppend(&result, "\n[MVSP: P0 selecting from ",
                      portfolio_p0_.size(), " strategies]");
      break;
    case Phase::kPortfolioP2:
      absl::StrAppend(&result, "\n[MVSP: P1 selecting from ",
                      portfolio_p1_.size(), " strategies]");
      break;
    case Phase::kMatrixTerminal:
      absl::StrAppend(&result, "\n[MVSP: Terminal, P0=",
                      mvs_selector_.P0Choice(), ", P1=",
                      mvs_selector_.P1Choice(), "]");
      break;
  }
  return result;
}

std::unique_ptr<State> MVSStateWithSubtreePureStrategies::Clone() const {
  return std::make_unique<MVSStateWithSubtreePureStrategies>(*this);
}

std::vector<std::pair<Action, double>>
MVSStateWithSubtreePureStrategies::ChanceOutcomes() const {
  if (selector_activated_ || AtDepthLimit()) return {};
  return state_->ChanceOutcomes();
}

void MVSStateWithSubtreePureStrategies::DoApplyAction(Action action_id) {
  if (selector_activated_) {
    mvs_selector_.ApplyAction(action_id);
    return;
  }

  if (AtDepthLimit()) {
    EnsurePortfoliosComputed();
    mvs_selector_.Activate(portfolio_p0_.size(), portfolio_p1_.size());
    selector_activated_ = true;
    mvs_selector_.ApplyAction(action_id);
    return;
  }

  bool was_chance = state_->IsChanceNode();
  state_->ApplyAction(action_id);

  if (!was_chance) {
    current_depth_++;
  } else if (GetMVSGame()->GetDepthMode() == MVSGame::DepthMode::kRoundBased) {
    current_round_++;
  }

  if (!state_->IsTerminal() && !state_->IsChanceNode() && AtDepthLimit()) {
    EnsurePortfoliosComputed();
    mvs_selector_.Activate(portfolio_p0_.size(), portfolio_p1_.size());
    selector_activated_ = true;
  }
}

std::shared_ptr<const MVSGameWithSubtreePureStrategies>
CreateMVSGameWithSubtreePureStrategies(
    std::shared_ptr<const Game> game,
    int depth_limit,
    MVSGame::DepthMode depth_mode,
    std::shared_ptr<Policy> opponent_model,
    Player opponent_player) {
  return std::make_shared<MVSGameWithSubtreePureStrategies>(
      std::move(game), depth_limit, depth_mode,
      std::move(opponent_model), opponent_player);
}

// ============================================================================
// MVS Utility Functions
// ============================================================================

// A policy wrapper that falls back to uniform when the primary policy has no
// entry at a given (state, player). Used to handle degenerate (e.g. p=1) LP
// solutions where some info states have zero reach and are missing.
class MVSUniformFallbackPolicy : public Policy {
 public:
  explicit MVSUniformFallbackPolicy(const Policy& primary) : primary_(primary) {}

  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override {
    auto result = primary_.GetStatePolicy(state, player);
    if (!result.empty()) return result;
    // Fallback: uniform over legal actions for this player.
    std::vector<Action> legal = state.LegalActions(player);
    if (legal.empty()) return {};
    ActionsAndProbs uniform;
    uniform.reserve(legal.size());
    double prob = 1.0 / legal.size();
    for (Action a : legal) uniform.push_back({a, prob});
    return uniform;
  }

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    return primary_.GetStatePolicy(info_state);
  }

 private:
  const Policy& primary_;
};

std::unordered_map<std::string, double> ExtractCFVsFromMVSSolution(
    const MVSGameWithSubtreePureStrategies& mvs_game,
    const Policy& mvs_policy,
    Player player) {
  std::unordered_map<std::string, double> cfvs;
  std::unordered_map<std::string, double> reach_sums;

  std::function<void(const State&, double, double)> traverse =
      [&](const State& state, double reach_p0, double reach_p1) {
    if (state.IsTerminal()) return;

    auto* mvs_state =
        dynamic_cast<const MVSStateWithSubtreePureStrategies*>(&state);
    if (mvs_state &&
        mvs_state->GetPhase() == MVSState::Phase::kPortfolioP1) {
      const State& underlying = mvs_state->GetUnderlyingState();
      std::string info_state = underlying.InformationStateString(player);

      MVSUniformFallbackPolicy mvs_fallback(mvs_policy);
      auto value = algorithms::ExpectedReturns(state, mvs_fallback, -1, false);

      double opponent_reach = (player == 0) ? reach_p1 : reach_p0;
      cfvs[info_state] += opponent_reach * value[player];
      reach_sums[info_state] += opponent_reach;
      return;
    }

    if (state.IsChanceNode()) {
      for (const auto& [action, prob] : state.ChanceOutcomes()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        traverse(*next, reach_p0 * prob, reach_p1 * prob);
      }
    } else {
      Player acting = state.CurrentPlayer();
      auto actions_probs = mvs_policy.GetStatePolicy(state, acting);
      if (actions_probs.empty()) {
        auto legal = state.LegalActions();
        double prob = 1.0 / legal.size();
        for (Action a : legal) {
          auto next = state.Clone();
          next->ApplyAction(a);
          double new_reach_p0 = reach_p0 * (acting == 0 ? prob : 1.0);
          double new_reach_p1 = reach_p1 * (acting == 1 ? prob : 1.0);
          traverse(*next, new_reach_p0, new_reach_p1);
        }
      } else {
        for (const auto& [action, prob] : actions_probs) {
          if (prob > 0) {
            auto next = state.Clone();
            next->ApplyAction(action);
            double new_reach_p0 = reach_p0 * (acting == 0 ? prob : 1.0);
            double new_reach_p1 = reach_p1 * (acting == 1 ? prob : 1.0);
            traverse(*next, new_reach_p0, new_reach_p1);
          }
        }
      }
    }
  };

  traverse(*mvs_game.NewInitialState(), 1.0, 1.0);
  return cfvs;
}

std::unordered_map<std::string, double> ExtractReachProbsFromMVS(
    const MVSGameWithSubtreePureStrategies& mvs_game,
    const Policy& mvs_policy,
    Player reaching_player) {
  std::unordered_map<std::string, double> reach_probs;

  std::function<void(const State&, double)> traverse =
      [&](const State& state, double reach) {
    if (state.IsTerminal()) return;

    auto* mvs_state =
        dynamic_cast<const MVSStateWithSubtreePureStrategies*>(&state);
    if (mvs_state &&
        mvs_state->GetPhase() == MVSState::Phase::kPortfolioP1) {
      const State& underlying = mvs_state->GetUnderlyingState();
      std::string hist = underlying.HistoryString();
      reach_probs[hist] = reach;
      return;
    }

    if (state.IsChanceNode()) {
      for (const auto& [action, prob] : state.ChanceOutcomes()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        traverse(*next, reach * prob);
      }
    } else {
      Player acting = state.CurrentPlayer();
      auto actions_probs = mvs_policy.GetStatePolicy(state, acting);
      if (actions_probs.empty()) {
        auto legal = state.LegalActions();
        double prob = 1.0 / legal.size();
        for (Action a : legal) {
          auto next = state.Clone();
          next->ApplyAction(a);
          double factor = (acting == reaching_player) ? prob : 1.0;
          traverse(*next, reach * factor);
        }
      } else {
        for (const auto& [action, prob] : actions_probs) {
          if (prob > 0) {
            auto next = state.Clone();
            next->ApplyAction(action);
            double factor = (acting == reaching_player) ? prob : 1.0;
            traverse(*next, reach * factor);
          }
        }
      }
    }
  };

  traverse(*mvs_game.NewInitialState(), 1.0);
  return reach_probs;
}

std::unordered_map<std::string, double> ExtractChanceReachFromMVS(
    const MVSGameWithSubtreePureStrategies& mvs_game) {
  std::unordered_map<std::string, double> chance_reach;

  std::function<void(const State&, double)> traverse =
      [&](const State& state, double reach) {
    if (state.IsTerminal()) return;

    auto* mvs_state =
        dynamic_cast<const MVSStateWithSubtreePureStrategies*>(&state);
    if (mvs_state &&
        mvs_state->GetPhase() == MVSState::Phase::kPortfolioP1) {
      const State& underlying = mvs_state->GetUnderlyingState();
      std::string hist = underlying.HistoryString();
      chance_reach[hist] = reach;
      return;
    }

    if (state.IsChanceNode()) {
      for (const auto& [action, prob] : state.ChanceOutcomes()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        traverse(*next, reach * prob);
      }
    } else {
      // Player actions don't contribute to chance reach
      for (Action a : state.LegalActions()) {
        auto next = state.Clone();
        next->ApplyAction(a);
        traverse(*next, reach);
      }
    }
  };

  traverse(*mvs_game.NewInitialState(), 1.0);
  return chance_reach;
}

}  // namespace open_spiel
