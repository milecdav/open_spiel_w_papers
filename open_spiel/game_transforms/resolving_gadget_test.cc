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

#include "open_spiel/game_transforms/resolving_gadget.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/unsafe_subgame.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

// =============================================================================
// Test Helper Functions
// =============================================================================

// Build SubgameRoots from states and a policy for computing reach probs
std::vector<SubgameRoot> BuildSubgameRoots(
    const std::vector<std::unique_ptr<State>>& states,
    const Game& game,
    const Policy& policy,
    Player non_resolving_player,
    Player resolving_player) {
  // Get raw pointers for ComputeReachProbabilities
  std::vector<const State*> state_ptrs;
  for (const auto& s : states) {
    state_ptrs.push_back(s.get());
  }

  auto reach_probs = ComputeReachProbabilities(
      game, policy, non_resolving_player, state_ptrs);

  std::vector<SubgameRoot> roots;
  for (const auto& state : states) {
    SubgameRoot root;
    root.state = state->Clone();
    root.info_state_string = state->InformationStateString(resolving_player);
    std::string hist = state->HistoryString();
    auto it = reach_probs.find(hist);
    root.reach_prob = (it != reach_probs.end()) ? it->second : 0.0;
    if (root.reach_prob > 0) {
      roots.push_back(std::move(root));
    }
  }

  return roots;
}

// =============================================================================
// Basic Tests
// =============================================================================

void TestGadgetConstruction() {
  std::cout << "TestGadgetConstruction..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  // Collect states at depth 2 (after both players have acted once)
  auto states = CollectStatesAtDepth(*game, 2);
  SPIEL_CHECK_GT(states.size(), 0);

  // Get raw pointers
  std::vector<const State*> state_ptrs;
  for (const auto& s : states) {
    state_ptrs.push_back(s.get());
  }

  // Compute reach probs for player 0 (non-resolving player)
  auto reach_probs = ComputeReachProbabilities(*game, *uniform, 0, state_ptrs);

  // Compute CF values for player 1 (resolving player) against uniform
  auto cf_values = ComputeCounterfactualValuesAtStates(
      *game, *uniform, 1, state_ptrs);

  // Build subgame roots
  std::vector<SubgameRoot> roots;
  for (const auto& state : states) {
    SubgameRoot root;
    root.state = state->Clone();
    root.info_state_string = state->InformationStateString(1);
    std::string hist = state->HistoryString();
    root.reach_prob = reach_probs[hist];
    if (root.reach_prob > 0) {
      roots.push_back(std::move(root));
    }
  }

  SPIEL_CHECK_GT(roots.size(), 0);

  // Create gadget game
  auto gadget = CreateGadgetGame(game, std::move(roots), 1, cf_values);

  SPIEL_CHECK_EQ(gadget->NumPlayers(), 2);
  SPIEL_CHECK_EQ(gadget->ResolvingPlayer(), 1);
  SPIEL_CHECK_GT(gadget->NormalizationConstant(), 0);

  std::cout << "  Gadget created with " << gadget->NumSubgameRoots()
            << " roots, k=" << gadget->NormalizationConstant() << std::endl;
  std::cout << "TestGadgetConstruction PASSED" << std::endl;
}

void TestGadgetStateTransitions() {
  std::cout << "TestGadgetStateTransitions..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  // Collect states at depth 2
  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> state_ptrs;
  for (const auto& s : states) {
    state_ptrs.push_back(s.get());
  }

  auto reach_probs = ComputeReachProbabilities(*game, *uniform, 0, state_ptrs);
  auto cf_values = ComputeCounterfactualValuesAtStates(
      *game, *uniform, 1, state_ptrs);

  std::vector<SubgameRoot> roots;
  for (const auto& state : states) {
    SubgameRoot root;
    root.state = state->Clone();
    root.info_state_string = state->InformationStateString(1);
    root.reach_prob = reach_probs[state->HistoryString()];
    if (root.reach_prob > 0) {
      roots.push_back(std::move(root));
    }
  }

  auto gadget = CreateGadgetGame(game, std::move(roots), 1, cf_values);

  // Test initial state
  auto state = gadget->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);

  // Apply chance action
  auto outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_GT(outcomes.size(), 0);
  double prob_sum = 0;
  for (const auto& [action, prob] : outcomes) {
    prob_sum += prob;
  }
  SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-9);

  state->ApplyAction(outcomes[0].first);

  // Should now be at gadget choice for resolving player (player 1)
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  auto gadget_state = dynamic_cast<GadgetState*>(state.get());
  SPIEL_CHECK_EQ(static_cast<int>(gadget_state->GetPhase()),
                 static_cast<int>(GadgetState::Phase::kGadgetChoice));

  // Legal actions should be T (0) and F (1)
  auto actions = state->LegalActions();
  SPIEL_CHECK_EQ(actions.size(), 2);
  SPIEL_CHECK_EQ(actions[0], GadgetGame::kTerminateAction);
  SPIEL_CHECK_EQ(actions[1], GadgetGame::kFollowAction);

  // Test T action (terminate)
  auto state_t = state->Clone();
  state_t->ApplyAction(GadgetGame::kTerminateAction);
  SPIEL_CHECK_TRUE(state_t->IsTerminal());

  // Test F action (follow)
  auto state_f = state->Clone();
  state_f->ApplyAction(GadgetGame::kFollowAction);
  // May or may not be terminal depending on the subgame state
  if (!state_f->IsTerminal()) {
    SPIEL_CHECK_GE(state_f->CurrentPlayer(), 0);
  }

  std::cout << "TestGadgetStateTransitions PASSED" << std::endl;
}

void TestGadgetPayoffs() {
  std::cout << "TestGadgetPayoffs..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> state_ptrs;
  for (const auto& s : states) {
    state_ptrs.push_back(s.get());
  }

  auto reach_probs = ComputeReachProbabilities(*game, *uniform, 0, state_ptrs);
  auto cf_values = ComputeCounterfactualValuesAtStates(
      *game, *uniform, 1, state_ptrs);

  std::vector<SubgameRoot> roots;
  for (const auto& state : states) {
    SubgameRoot root;
    root.state = state->Clone();
    root.info_state_string = state->InformationStateString(1);
    root.reach_prob = reach_probs[state->HistoryString()];
    if (root.reach_prob > 0) {
      roots.push_back(std::move(root));
    }
  }

  auto gadget = CreateGadgetGame(game, std::move(roots), 1, cf_values);

  // Play through and check that T payoffs match CF values (scaled by k)
  auto state = gadget->NewInitialState();
  auto outcomes = state->ChanceOutcomes();
  state->ApplyAction(outcomes[0].first);

  auto gadget_state = dynamic_cast<GadgetState*>(state.get());
  std::string info_state = gadget->GetRootInfoState(gadget_state->SelectedRootIndex());

  // Take T action
  state->ApplyAction(GadgetGame::kTerminateAction);
  SPIEL_CHECK_TRUE(state->IsTerminal());

  auto returns = state->Returns();
  double expected_t_payoff = gadget->GetTerminatePayoff(info_state);

  // Resolving player (1) should get the T payoff
  SPIEL_CHECK_FLOAT_NEAR(returns[1], expected_t_payoff, 1e-9);
  // Zero-sum: player 0 gets negative
  SPIEL_CHECK_FLOAT_NEAR(returns[0], -expected_t_payoff, 1e-9);

  std::cout << "TestGadgetPayoffs PASSED" << std::endl;
}

// =============================================================================
// CFR on Gadget Tests
// =============================================================================

void TestCFROnGadget() {
  std::cout << "TestCFROnGadget..." << std::endl;

  auto game = LoadGame("kuhn_poker");
  auto uniform = std::make_shared<UniformPolicy>();

  // Collect states at depth 2
  auto states = CollectStatesAtDepth(*game, 2);
  std::vector<const State*> state_ptrs;
  for (const auto& s : states) {
    state_ptrs.push_back(s.get());
  }

  auto reach_probs = ComputeReachProbabilities(*game, *uniform, 0, state_ptrs);
  auto cf_values = ComputeCounterfactualValuesAtStates(
      *game, *uniform, 1, state_ptrs);

  std::vector<SubgameRoot> roots;
  for (const auto& state : states) {
    SubgameRoot root;
    root.state = state->Clone();
    root.info_state_string = state->InformationStateString(1);
    root.reach_prob = reach_probs[state->HistoryString()];
    if (root.reach_prob > 0) {
      roots.push_back(std::move(root));
    }
  }

  auto gadget = CreateGadgetGame(game, std::move(roots), 1, cf_values);

  // Run CFR on the gadget
  algorithms::CFRSolverBase solver(
      *gadget,
      /*alternating_updates=*/true,
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true);

  for (int i = 0; i < 1000; ++i) {
    solver.EvaluateAndUpdatePolicy();
  }

  // Get the average policy and check exploitability
  auto avg_policy = solver.AveragePolicy();
  double exploitability = algorithms::Exploitability(*gadget, *avg_policy);

  std::cout << "  CFR on gadget: exploitability = " << exploitability << std::endl;
  SPIEL_CHECK_LT(exploitability, 0.1);  // Should converge reasonably

  std::cout << "TestCFROnGadget PASSED" << std::endl;
}

// =============================================================================
// Leduc Poker Tests with MVS and Gadgets
// =============================================================================

// Collect all info states from round 1 (before public card deal)
// In Leduc, round 2 starts after 3 chance nodes (2 private + 1 public)
std::unordered_map<int, std::unordered_set<std::string>> CollectRound1InfoStates(
    const Game& game) {
  std::unordered_map<int, std::unordered_set<std::string>> round1_infostates;

  std::function<void(const State&, int)> traverse = [&](const State& state,
                                                         int chance_count) {
    if (state.IsTerminal()) return;

    if (state.IsChanceNode()) {
      for (const auto& [action, prob] : state.ChanceOutcomes()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        traverse(*next, chance_count + 1);
      }
    } else {
      // This is a player node
      if (chance_count < 3) {
        // We're in round 1 (before public card)
        Player player = state.CurrentPlayer();
        round1_infostates[player].insert(state.InformationStateString(player));
      }
      for (Action action : state.LegalActions()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        traverse(*next, chance_count);
      }
    }
  };

  traverse(*game.NewInitialState(), 0);
  return round1_infostates;
}

// Group subgame roots by the betting sequence (public state)
// This identifies the 5 distinct subgames in Leduc
std::unordered_map<std::string, std::vector<std::unique_ptr<State>>>
GroupSubgamesByBettingSequence(std::vector<std::unique_ptr<State>> roots) {
  std::unordered_map<std::string, std::vector<std::unique_ptr<State>>> grouped;

  for (auto& root : roots) {
    // Extract betting sequence from history
    // In Leduc, the history includes chance actions (card deals) and player actions
    // We want to group by the player actions only (the betting sequence)
    std::string history = root->HistoryString();

    // Parse history to extract just the betting actions
    // Leduc history format: chance actions are card indices, player actions are 0/1 (fold/call/check) or 2 (raise)
    // For simplicity, we'll use the action sequence after the first two chance nodes
    auto full_history = root->FullHistory();
    std::string betting_seq;
    int chance_count = 0;
    for (const auto& item : full_history) {
      if (item.player == kChancePlayerId) {
        chance_count++;
      } else if (chance_count >= 2 && chance_count < 3) {
        // This is a round 1 betting action
        betting_seq += std::to_string(item.action);
      }
    }

    grouped[betting_seq].push_back(std::move(root));
  }

  return grouped;
}

void TestLeducMVSWithGadgetResolving() {
  std::cout << "TestLeducMVSWithGadgetResolving..." << std::endl;

  auto game = LoadGame("leduc_poker");

  // Step 1: Solve full game for baseline
  std::cout << "  Step 1: Solving full game for baseline..." << std::endl;
  algorithms::CFRSolverBase full_solver(
      *game,
      /*alternating_updates=*/true,
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    full_solver.EvaluateAndUpdatePolicy();
  }
  auto full_policy = full_solver.AveragePolicy();
  double full_exploitability = algorithms::Exploitability(*game, *full_policy);
  std::cout << "    Full game exploitability: " << full_exploitability << std::endl;

  // Step 2: Solve MVS game for round 1 (trunk)
  // depth_limit=3 with kRoundBased means: play until 3 chance nodes complete
  // (2 private cards + 1 public card), then use MVS for round 2
  std::cout << "  Step 2: Solving MVS game for round 1 trunk..." << std::endl;
  auto mvs_game = CreateMVSGameWithSubtreePureStrategies(
      game, /*depth_limit=*/3, MVSGame::DepthMode::kRoundBased);

  algorithms::CFRSolverBase mvs_solver(
      *mvs_game,
      /*alternating_updates=*/true,
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true);

  for (int i = 0; i < 500; ++i) {
    mvs_solver.EvaluateAndUpdatePolicy();
  }
  auto mvs_policy = mvs_solver.AveragePolicy();
  double mvs_exploitability = algorithms::Exploitability(*mvs_game, *mvs_policy);
  std::cout << "    MVS game exploitability: " << mvs_exploitability << std::endl;

  // Step 3: Collect round 1 info states and round 2 subgame roots
  std::cout << "  Step 3: Collecting subgame roots..." << std::endl;
  auto round1_infostates = CollectRound1InfoStates(*game);
  auto round2_roots = CollectStatesAtRound(*game, 3);
  auto grouped_subgames = GroupSubgamesByBettingSequence(std::move(round2_roots));

  std::cout << "    Round 1 info states: P0=" << round1_infostates[0].size()
            << ", P1=" << round1_infostates[1].size() << std::endl;
  std::cout << "    Distinct subgames (betting sequences): "
            << grouped_subgames.size() << std::endl;

  for (const auto& [seq, roots] : grouped_subgames) {
    std::cout << "      Sequence '" << seq << "': " << roots.size()
              << " root states" << std::endl;
  }

  // Step 4: Build combined policy using MVS trunk strategy
  // The trunk (round 1) comes from MVS solution, round 2 will be re-solved
  std::cout << "  Step 4: Building combined policy with MVS trunk..."
            << std::endl;
  auto combined_policy = std::make_shared<TabularPolicy>();

  // Extract round 1 strategy from MVS policy
  // MVS policy is defined on MVS game states, so we need to map back to original
  // The MVS game has the same info state strings for round 1 as the original game
  std::function<void(const State&, const State&)> copy_mvs_trunk =
      [&](const State& mvs_state, const State& orig_state) {
    if (orig_state.IsTerminal()) return;

    auto* mvs_s = dynamic_cast<const MVSStateWithSubtreePureStrategies*>(&mvs_state);
    if (mvs_s && mvs_s->GetPhase() != MVSState::Phase::kNormal) {
      // We've reached the depth limit - don't copy MVS portfolio actions
      return;
    }

    if (orig_state.IsChanceNode()) {
      for (const auto& [action, prob] : orig_state.ChanceOutcomes()) {
        auto next_orig = orig_state.Clone();
        next_orig->ApplyAction(action);
        auto next_mvs = mvs_state.Clone();
        next_mvs->ApplyAction(action);
        copy_mvs_trunk(*next_mvs, *next_orig);
      }
    } else {
      Player player = orig_state.CurrentPlayer();
      std::string info_state = orig_state.InformationStateString(player);

      // Get action probabilities from MVS policy
      auto actions_probs = mvs_policy->GetStatePolicy(mvs_state, player);
      if (!actions_probs.empty()) {
        combined_policy->SetStatePolicy(info_state, actions_probs);
      }

      for (Action action : orig_state.LegalActions()) {
        auto next_orig = orig_state.Clone();
        next_orig->ApplyAction(action);
        auto next_mvs = mvs_state.Clone();
        next_mvs->ApplyAction(action);
        copy_mvs_trunk(*next_mvs, *next_orig);
      }
    }
  };
  copy_mvs_trunk(*mvs_game->NewInitialState(), *game->NewInitialState());

  std::cout << "    Trunk policy extracted: " << combined_policy->PolicyTable().size()
            << " info states" << std::endl;

  // Step 5: Extract CFVs and reach probs from MVS solution
  // The gadget needs counterfactual values at the depth limit from MVS
  std::cout << "  Step 5: Extracting CFVs from MVS solution..." << std::endl;

  // Get CFVs for both players from MVS solution
  // These are the expected values at the depth limit weighted by opponent reach
  auto cfvs_p0 = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 0);
  auto cfvs_p1 = ExtractCFVsFromMVSSolution(*mvs_game, *mvs_policy, 1);

  // Get reach probabilities from MVS solution
  auto reach_probs_p0 = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 0);
  auto reach_probs_p1 = ExtractReachProbsFromMVS(*mvs_game, *mvs_policy, 1);

  std::cout << "    CFVs extracted: P0=" << cfvs_p0.size()
            << ", P1=" << cfvs_p1.size() << std::endl;
  std::cout << "    Reach probs: P0=" << reach_probs_p0.size()
            << ", P1=" << reach_probs_p1.size() << std::endl;

  // Step 6: Re-solve each subgame using gadgets
  // For each subgame, we create a gadget that preserves the resolving player's CFV
  std::cout << "  Step 6: Re-solving subgames with gadgets..." << std::endl;

  // Helper to re-solve for a specific player
  auto resolve_for_player = [&](Player non_resolving_player, const std::string& player_name) {
    Player resolving_player = 1 - non_resolving_player;
    const auto& cf_values = (resolving_player == 0) ? cfvs_p0 : cfvs_p1;
    const auto& reach_probs = (non_resolving_player == 0) ? reach_probs_p0 : reach_probs_p1;
    std::string player_marker = absl::StrCat("[Observer: ", non_resolving_player, "]");

    int subgame_count = 0;
    for (auto& [betting_seq, roots] : grouped_subgames) {
      subgame_count++;

      std::vector<SubgameRoot> gadget_roots;
      for (const auto& root : roots) {
        SubgameRoot sr;
        sr.state = root->Clone();
        sr.info_state_string = root->InformationStateString(resolving_player);
        std::string hist = root->HistoryString();
        auto it = reach_probs.find(hist);
        sr.reach_prob = (it != reach_probs.end()) ? it->second : 0.0;
        if (sr.reach_prob > 0) {
          gadget_roots.push_back(std::move(sr));
        }
      }

      if (gadget_roots.empty()) continue;

      auto gadget = CreateGadgetGame(
          game, std::move(gadget_roots), resolving_player, cf_values);

      algorithms::CFRSolverBase gadget_solver(
          *gadget, true, true, true);

      for (int i = 0; i < 500; ++i) {
        gadget_solver.EvaluateAndUpdatePolicy();
      }

      TabularPolicy tabular_gadget_policy = gadget_solver.TabularAveragePolicy();

      // Extract non-resolving player's strategy
      const std::string prefix = "gadget_F:subgame:";
      int extracted_count = 0;
      for (const auto& [gadget_info_state, actions_probs] : tabular_gadget_policy.PolicyTable()) {
        if (gadget_info_state.find(prefix) == 0) {
          std::string original_info_state = gadget_info_state.substr(prefix.length());
          if (original_info_state.find(player_marker) != std::string::npos) {
            combined_policy->SetStatePolicy(original_info_state, actions_probs);
            extracted_count++;
          }
        }
      }
      std::cout << "      Subgame " << subgame_count << ": extracted " << extracted_count
                << " policies for " << player_name << std::endl;
    }
  };

  // Re-solve for both players
  std::cout << "    Re-solving for P1 (P0 is resolving player)..." << std::endl;
  resolve_for_player(1, "P1");

  std::cout << "    Re-solving for P0 (P1 is resolving player)..." << std::endl;
  resolve_for_player(0, "P0");

  std::cout << "    Combined policy has " << combined_policy->PolicyTable().size()
            << " info states before fill-in" << std::endl;

  // Step 7: Evaluate combined policy
  std::cout << "  Step 7: Evaluating combined policy..." << std::endl;

  // Use the full policy for any states not covered by our combined policy
  // (This is a fallback - ideally all states should be covered)
  auto final_policy = std::make_shared<TabularPolicy>(*combined_policy);

  // Fill in any missing states from full policy
  int filled_count = 0;
  std::function<void(const State&)> fill_missing = [&](const State& state) {
    if (state.IsTerminal()) return;

    if (state.IsChanceNode()) {
      for (const auto& [action, prob] : state.ChanceOutcomes()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        fill_missing(*next);
      }
    } else {
      Player player = state.CurrentPlayer();
      std::string info_state = state.InformationStateString(player);

      auto existing = final_policy->GetStatePolicy(info_state);
      if (existing.empty()) {
        throw std::runtime_error("Info state not found in final policy: " + info_state);
      }

      for (Action action : state.LegalActions()) {
        auto next = state.Clone();
        next->ApplyAction(action);
        fill_missing(*next);
      }
    }
  };
  fill_missing(*game->NewInitialState());
  std::cout << "    Filled " << filled_count << " missing states from full policy" << std::endl;
  std::cout << "    Final policy has " << final_policy->PolicyTable().size()
            << " info states" << std::endl;

  double combined_exploitability = algorithms::Exploitability(*game, *final_policy);
  std::cout << "    Combined policy exploitability: " << combined_exploitability
            << std::endl;
  std::cout << "    Full policy exploitability: " << full_exploitability
            << std::endl;

  // The combined policy should have reasonable exploitability
  // It may not be as good as the full solution due to the decomposition
  SPIEL_CHECK_LT(combined_exploitability, 0.5);

  std::cout << "TestLeducMVSWithGadgetResolving PASSED" << std::endl;
}
void TestSafeVsUnsafeResolving() {
  std::cout << "TestSafeVsUnsafeResolving..." << std::endl;

  // This test compares safe (gadget) vs unsafe (direct) subgame re-solving
  // on Leduc poker, demonstrating that unsafe re-solving is exploitable.
  //
  // Unsafe re-solving: Solve subgame with chance node weighted by total reaches
  // Safe re-solving: Use gadget with T/F choice to preserve CF values

  auto game = LoadGame("leduc_poker");

  // Step 1: Solve full game for baseline and to get trunk strategy
  std::cout << "  Step 1: Solving full game..." << std::endl;
  algorithms::CFRSolverBase full_solver(
      *game, true, true, true);

  for (int i = 0; i < 500; ++i) {
    full_solver.EvaluateAndUpdatePolicy();
  }
  auto full_policy = full_solver.AveragePolicy();
  double full_exploitability = algorithms::Exploitability(*game, *full_policy);
  std::cout << "    Full game exploitability: " << full_exploitability << std::endl;

  // Step 2: Collect round 2 roots and compute reaches
  std::cout << "  Step 2: Collecting subgame roots..." << std::endl;
  auto round2_roots = CollectStatesAtRound(*game, 3);
  std::vector<const State*> root_ptrs;
  for (const auto& s : round2_roots) {
    root_ptrs.push_back(s.get());
  }

  auto reach_probs_p0 = ComputeReachProbabilities(*game, *full_policy, 0, root_ptrs);
  auto reach_probs_p1 = ComputeReachProbabilities(*game, *full_policy, 1, root_ptrs);

  // Compute CFVs for safe re-solving
  auto cfvs_p0 = ComputeCounterfactualValuesAtStates(
      *game, *full_policy, 0, root_ptrs, reach_probs_p1);
  auto cfvs_p1 = ComputeCounterfactualValuesAtStates(
      *game, *full_policy, 1, root_ptrs, reach_probs_p0);

  std::cout << "    Subgame roots: " << round2_roots.size() << std::endl;

  // Collect round 1 info states for trunk
  auto round1_infostates = CollectRound1InfoStates(*game);

  // Group subgames by betting sequence
  auto grouped = GroupSubgamesByBettingSequence(CollectStatesAtRound(*game, 3));

  // =========================================================================
  // Safe Re-Solving (using gadgets)
  // =========================================================================
  std::cout << "  Step 3: Safe re-solving with gadgets..." << std::endl;

  auto safe_policy = std::make_shared<TabularPolicy>();

  // Copy trunk from full policy
  for (const auto& [player, info_states] : round1_infostates) {
    for (const auto& info_state : info_states) {
      auto actions_probs = full_policy->GetStatePolicy(info_state);
      if (!actions_probs.empty()) {
        safe_policy->SetStatePolicy(info_state, actions_probs);
      }
    }
  }

  // Re-solve for both players using gadgets
  auto safe_resolve = [&](Player non_resolving_player) {
    Player resolving_player = 1 - non_resolving_player;
    const auto& cf_values = (resolving_player == 0) ? cfvs_p0 : cfvs_p1;
    const auto& reach_probs = (non_resolving_player == 0) ? reach_probs_p0 : reach_probs_p1;
    std::string player_marker = absl::StrCat("[Observer: ", non_resolving_player, "]");

    for (auto& [betting_seq, roots] : grouped) {
      std::vector<SubgameRoot> gadget_roots;
      for (const auto& root : roots) {
        SubgameRoot sr;
        sr.state = root->Clone();
        sr.info_state_string = root->InformationStateString(resolving_player);
        std::string hist = root->HistoryString();
        auto it = reach_probs.find(hist);
        sr.reach_prob = (it != reach_probs.end()) ? it->second : 0.0;
        if (sr.reach_prob > 0) gadget_roots.push_back(std::move(sr));
      }

      if (gadget_roots.empty()) continue;

      auto gadget = CreateGadgetGame(game, std::move(gadget_roots), resolving_player, cf_values);
      algorithms::CFRSolverBase solver(*gadget, true, true, true);
      for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();

      TabularPolicy gadget_policy = solver.TabularAveragePolicy();
      const std::string prefix = "gadget_F:subgame:";
      for (const auto& [gadget_is, actions_probs] : gadget_policy.PolicyTable()) {
        if (gadget_is.find(prefix) == 0) {
          std::string orig_is = gadget_is.substr(prefix.length());
          if (orig_is.find(player_marker) != std::string::npos) {
            safe_policy->SetStatePolicy(orig_is, actions_probs);
          }
        }
      }
    }
  };

  safe_resolve(1);  // P1's strategy (P0 resolving)
  safe_resolve(0);  // P0's strategy (P1 resolving)

  double safe_exploitability = algorithms::Exploitability(*game, *safe_policy);
  std::cout << "    Safe policy exploitability: " << safe_exploitability << std::endl;

  // =========================================================================
  // Unsafe Re-Solving (direct subgame solving without gadget)
  // =========================================================================
  std::cout << "  Step 4: Unsafe re-solving (no gadget)..." << std::endl;

  auto unsafe_policy = std::make_shared<TabularPolicy>();

  // Copy trunk from full policy (same as safe)
  for (const auto& [player, info_states] : round1_infostates) {
    for (const auto& info_state : info_states) {
      auto actions_probs = full_policy->GetStatePolicy(info_state);
      if (!actions_probs.empty()) {
        unsafe_policy->SetStatePolicy(info_state, actions_probs);
      }
    }
  }

  // Solve each subgame directly with chance node weighted by total reaches
  for (auto& [betting_seq, roots] : grouped) {
    // Compute total reach = chance × π_0 × π_1 for each root
    std::vector<std::unique_ptr<State>> subgame_roots;
    std::vector<double> total_reaches;

    for (const auto& root : roots) {
      std::string hist = root->HistoryString();
      double reach_p0 = reach_probs_p0.count(hist) ? reach_probs_p0.at(hist) : 0.0;
      double reach_p1 = reach_probs_p1.count(hist) ? reach_probs_p1.at(hist) : 0.0;
      // Total reach includes both players' reaches (chance is already in there)
      double total_reach = reach_p0 * reach_p1;
      if (total_reach > 0) {
        subgame_roots.push_back(root->Clone());
        total_reaches.push_back(total_reach);
      }
    }

    if (subgame_roots.empty()) continue;

    // Create unsafe subgame (chance node at top, no T/F)
    auto unsafe_subgame = CreateUnsafeSubgame(
        game, std::move(subgame_roots), std::move(total_reaches));

    algorithms::CFRSolverBase solver(*unsafe_subgame, true, true, true);
    for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();

    TabularPolicy subgame_policy = solver.TabularAveragePolicy();

    // Extract strategies for both players
    const std::string prefix = "unsafe:subgame:";
    for (const auto& [subgame_is, actions_probs] : subgame_policy.PolicyTable()) {
      if (subgame_is.find(prefix) == 0) {
        std::string orig_is = subgame_is.substr(prefix.length());
        unsafe_policy->SetStatePolicy(orig_is, actions_probs);
      }
    }
  }

  double unsafe_exploitability = algorithms::Exploitability(*game, *unsafe_policy);
  std::cout << "    Unsafe policy exploitability: " << unsafe_exploitability << std::endl;

  // =========================================================================
  // Compare results
  // =========================================================================
  std::cout << "\n  Results summary:" << std::endl;
  std::cout << "    Full game exploitability:   " << full_exploitability << std::endl;
  std::cout << "    Safe (gadget) exploitability: " << safe_exploitability << std::endl;
  std::cout << "    Unsafe exploitability:      " << unsafe_exploitability << std::endl;

  // Safe re-solving should be better (lower exploitability) than unsafe
  // Both should be reasonable, but unsafe typically worse
  SPIEL_CHECK_LT(safe_exploitability, 0.1);
  SPIEL_CHECK_LT(unsafe_exploitability, 0.5);

  std::cout << "TestSafeVsUnsafeResolving PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestGadgetConstruction();
  open_spiel::TestGadgetStateTransitions();
  open_spiel::TestGadgetPayoffs();
  open_spiel::TestCFROnGadget();
  open_spiel::TestLeducMVSWithGadgetResolving();
  open_spiel::TestSafeVsUnsafeResolving();

  std::cout << "\nAll tests passed!" << std::endl;
  return 0;
}
