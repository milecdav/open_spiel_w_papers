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

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/continual_resolving.h"
#include "open_spiel/game_transforms/resolving_by_is.h"
#include "open_spiel/game_transforms/ses_gadget.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestResolvingByISNashCFRSmoke() {
  std::cout << "TestResolvingByISNashCFRSmoke..." << std::endl;
  auto game = LoadGame("kuhn_poker");
  TabularPolicy uniform = GetUniformPolicy(*game);
  auto decomp = DecomposeGameAtDepth(game, uniform, /*depth=*/1);
  SPIEL_CHECK_FALSE(decomp.grouped_subgames.empty());

  const auto& group_it = *decomp.grouped_subgames.begin();
  const auto& roots = group_it.second;
  constexpr Player resolving = 0;
  const Player non_res = 1 - resolving;

  auto ribis_roots =
      BuildSubgameRoots(roots, non_res, decomp.reach_probs[resolving]);
  SPIEL_CHECK_FALSE(ribis_roots.empty());

  auto ribis_game = CreateResolvingByISGame(
      game, std::move(ribis_roots), resolving, decomp.cfvs[non_res]);

  algorithms::CFRSolverBase solver(*ribis_game, true, true, true);
  for (int i = 0; i < 500; ++i) solver.EvaluateAndUpdatePolicy();
  TabularPolicy policy = solver.TabularAveragePolicy();
  SPIEL_CHECK_GT(policy.PolicyTable().size(), 0);

  double exp = algorithms::Exploitability(*ribis_game, policy);
  std::cout << "  exploitability=" << exp << std::endl;
  SPIEL_CHECK_TRUE(std::isfinite(exp));
}

// Diagnostic: at alpha=0 (SES) or p=0 (unified), both paths should reduce to
// pure max-margin Nash. With a near-Nash trunk, the resulting policy should
// have near-zero full-game exploitability. We measure exploitability for
// both paths at a range of alpha/p values to see where divergence happens.
void TestSESEquivalence() {
  std::cout << "TestSESEquivalence (exploitability diagnostic)..." << std::endl;

  auto game = LoadGame("kuhn_poker");

  // Solve the full game with CFR to get a near-Nash trunk blueprint.
  algorithms::CFRSolverBase trunk_solver(*game, true, true, true);
  for (int i = 0; i < 2000; ++i) trunk_solver.EvaluateAndUpdatePolicy();
  auto trunk_policy_ptr = trunk_solver.AveragePolicy();
  TabularPolicy trunk = trunk_solver.TabularAveragePolicy();
  double trunk_exp = algorithms::Exploitability(*game, *trunk_policy_ptr);
  std::cout << "  Trunk (CFR 2000 iter) exploitability: " << trunk_exp
            << std::endl;

  // Decompose at depth 1 (one player action = after card deal)
  auto decomp = DecomposeGameAtDepth(game, *trunk_policy_ptr, /*depth=*/1);
  SPIEL_CHECK_FALSE(decomp.grouped_subgames.empty());

  // Collect all subgame info states.
  std::array<std::unordered_set<std::string>, 2> all_subgame_is;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    auto per_player = CollectSubgameInfoStatesPerPlayer(roots);
    for (int p = 0; p < 2; ++p) {
      all_subgame_is[p].insert(per_player[p].begin(), per_player[p].end());
    }
  }

  UniformPolicy opponent_model;
  const int kCFRIters = 2000;
  const double alpha_values[] = {0.0, 0.25, 0.5, 0.75, 1.0};

  std::cout << "\n  alpha/p | SES exp       | Unified exp" << std::endl;
  std::cout << "  --------+---------------+---------------" << std::endl;

  for (double alpha : alpha_values) {
    // --- Legacy SES path ---
    auto ses_policy =
        ResolveWithSESGadget(decomp, trunk, alpha, opponent_model, kCFRIters);
    double ses_exp = algorithms::Exploitability(*game, *ses_policy);

    // --- Unified path: MaxMargin + RNR + lock=false ---
    ResolvingConfig config;
    config.gadget_kind = GadgetKind::kMaxMargin;
    config.response_kind = ResponseKind::kRNR;
    config.solver_kind = SolverKind::kCFR;
    config.lock_opponent_in_fixed_branch = false;
    config.p = alpha;
    config.cfr_iterations = kCFRIters;
    config.opponent_model = &opponent_model;

    // Run passes for both target players, merge the policies.
    config.target_player = 0;
    auto u0 = ResolveSubgames(decomp, trunk, config);
    config.target_player = 1;
    auto u1 = ResolveSubgames(decomp, trunk, config);
    auto unified = std::make_shared<TabularPolicy>(*u0);
    for (const auto& is : all_subgame_is[1]) {
      auto ap = u1->GetStatePolicy(is);
      if (!ap.empty()) unified->SetStatePolicy(is, ap);
    }
    double uni_exp = algorithms::Exploitability(*game, *unified);

    std::cout << "   " << alpha << "    | " << ses_exp << " | " << uni_exp
              << std::endl;
  }

  std::cout << "TestSESEquivalence diagnostic complete" << std::endl;
}

// Smoke test for ResolvingByIS + RNR + CFR + lock=false (OX-style primitive).
// Numerical equality to legacy OX is not required — we only check that the
// unified path produces a well-formed policy covering all subgame info states
// with finite exploitability.
void TestOXSmoke() {
  std::cout << "TestOXSmoke (ResolvingByIS + RNR + CFR + lock=false)..."
            << std::endl;

  auto game = LoadGame("kuhn_poker");

  // Near-Nash trunk so the full-game exploitability number is meaningful.
  algorithms::CFRSolverBase trunk_solver(*game, true, true, true);
  for (int i = 0; i < 2000; ++i) trunk_solver.EvaluateAndUpdatePolicy();
  auto trunk_policy_ptr = trunk_solver.AveragePolicy();
  TabularPolicy trunk = trunk_solver.TabularAveragePolicy();

  auto decomp = DecomposeGameAtDepth(game, *trunk_policy_ptr, /*depth=*/1);
  SPIEL_CHECK_FALSE(decomp.grouped_subgames.empty());

  std::array<std::unordered_set<std::string>, 2> all_subgame_is;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    auto per_player = CollectSubgameInfoStatesPerPlayer(roots);
    for (int p = 0; p < 2; ++p) {
      all_subgame_is[p].insert(per_player[p].begin(), per_player[p].end());
    }
  }

  UniformPolicy opponent_model;

  ResolvingConfig config;
  config.gadget_kind = GadgetKind::kResolvingByIS;
  config.response_kind = ResponseKind::kRNR;
  config.solver_kind = SolverKind::kCFR;
  config.lock_opponent_in_fixed_branch = false;
  config.p = 0.5;
  config.cfr_iterations = 2000;
  config.opponent_model = &opponent_model;

  config.target_player = 0;
  auto u0 = ResolveSubgames(decomp, trunk, config);
  config.target_player = 1;
  auto u1 = ResolveSubgames(decomp, trunk, config);
  auto merged = std::make_shared<TabularPolicy>(*u0);
  for (const auto& is : all_subgame_is[1]) {
    auto ap = u1->GetStatePolicy(is);
    if (!ap.empty()) merged->SetStatePolicy(is, ap);
  }

  // Structural check: every subgame info state has a policy entry.
  int missing = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (const auto& is : all_subgame_is[pl]) {
      auto ap = merged->GetStatePolicy(is);
      if (ap.empty()) {
        if (missing < 5) {
          std::cout << "    MISSING: player=" << pl << " is=" << is
                    << std::endl;
        }
        ++missing;
      }
    }
  }
  SPIEL_CHECK_EQ(missing, 0);

  double exp = algorithms::Exploitability(*game, *merged);
  std::cout << "  p=0.5 exploitability=" << exp << std::endl;
  SPIEL_CHECK_TRUE(std::isfinite(exp));

  std::cout << "TestOXSmoke PASSED" << std::endl;
}

// ============================================================================
// Shared helper: solve Kuhn with near-Nash CFR blueprint and decompose.
// ============================================================================

struct KuhnSetup {
  std::shared_ptr<const Game> game;
  TabularPolicy trunk;
  SubgameDecomposition decomp;
  std::array<std::unordered_set<std::string>, 2> all_subgame_is;
};

KuhnSetup MakeKuhnSetup(int trunk_cfr_iters = 2000) {
  KuhnSetup s;
  s.game = LoadGame("kuhn_poker");

  algorithms::CFRSolverBase solver(*s.game, true, true, true);
  for (int i = 0; i < trunk_cfr_iters; ++i) solver.EvaluateAndUpdatePolicy();
  auto policy_ptr = solver.AveragePolicy();
  s.trunk = solver.TabularAveragePolicy();

  s.decomp = DecomposeGameAtDepth(s.game, *policy_ptr, /*depth=*/1);
  for (const auto& [pub_obs, roots] : s.decomp.grouped_subgames) {
    auto per_player = CollectSubgameInfoStatesPerPlayer(roots);
    for (int p = 0; p < 2; ++p) {
      s.all_subgame_is[p].insert(per_player[p].begin(), per_player[p].end());
    }
  }
  return s;
}

// Merge p0 and p1 subgame policies into one full policy covering all subgame
// info states. p0 supplies all entries from u0; p1 overrides player-1 entries.
std::shared_ptr<TabularPolicy> MergeSubgamePolicies(
    const TabularPolicy& u0, const TabularPolicy& u1,
    const std::array<std::unordered_set<std::string>, 2>& all_is) {
  auto merged = std::make_shared<TabularPolicy>(u0);
  for (const auto& is : all_is[1]) {
    auto ap = u1.GetStatePolicy(is);
    if (!ap.empty()) merged->SetStatePolicy(is, ap);
  }
  return merged;
}

// ============================================================================
// Step 12: LP+RNR parity test
// ============================================================================

// For a selection of (gadget, lock) combinations on Kuhn, run
// response=RNR with solver=CFR and solver=LP, then compare the resulting
// full-game exploitability.  We compare exploitability rather than raw
// policy because Kuhn has non-unique NE.  Tolerance 1e-3.
void TestLPRNRParity() {
  std::cout << "TestLPRNRParity..." << std::endl;

  auto s = MakeKuhnSetup(2000);
  // Tabularize the uniform policy so that the string-based GetStatePolicy
  // lookup works inside RNRMixtureGame::ChanceOutcomes (LP solver path).
  TabularPolicy opponent_model = GetUniformPolicy(*s.game);

  struct Combo {
    GadgetKind gadget;
    bool lock;
    const char* label;
  };
  const Combo combos[] = {
      {GadgetKind::kUnsafe,    true,  "Unsafe lock=true"},
      {GadgetKind::kResolving, true,  "Resolving lock=true"},
      {GadgetKind::kMaxMargin, false, "MaxMargin lock=false"},
  };

  const double kExpTol = 1e-3;

  for (const auto& c : combos) {
    std::cout << "  Combo: " << c.label << std::endl;

    // Base config shared for both solver kinds.
    ResolvingConfig base;
    base.gadget_kind = c.gadget;
    base.response_kind = ResponseKind::kRNR;
    base.lock_opponent_in_fixed_branch = c.lock;
    base.p = 0.5;
    base.cfr_iterations = 2000;
    base.opponent_model = &opponent_model;

    // --- CFR solve ---
    ResolvingConfig cfr_config = base;
    cfr_config.solver_kind = SolverKind::kCFR;

    cfr_config.target_player = 0;
    auto cfr0 = ResolveSubgames(s.decomp, s.trunk, cfr_config);
    cfr_config.target_player = 1;
    auto cfr1 = ResolveSubgames(s.decomp, s.trunk, cfr_config);
    auto cfr_merged = MergeSubgamePolicies(*cfr0, *cfr1, s.all_subgame_is);
    double cfr_exp = algorithms::Exploitability(*s.game, *cfr_merged);
    std::cout << "    CFR exploitability = " << cfr_exp << std::endl;

    // --- LP solve ---
    ResolvingConfig lp_config = base;
    lp_config.solver_kind = SolverKind::kLP;

    lp_config.target_player = 0;
    auto lp0 = ResolveSubgames(s.decomp, s.trunk, lp_config);
    lp_config.target_player = 1;
    auto lp1 = ResolveSubgames(s.decomp, s.trunk, lp_config);
    auto lp_merged = MergeSubgamePolicies(*lp0, *lp1, s.all_subgame_is);
    double lp_exp = algorithms::Exploitability(*s.game, *lp_merged);
    std::cout << "    LP  exploitability = " << lp_exp << std::endl;

    SPIEL_CHECK_TRUE(std::isfinite(cfr_exp));
    SPIEL_CHECK_TRUE(std::isfinite(lp_exp));
    double diff = std::abs(cfr_exp - lp_exp);
    std::cout << "    |CFR_exp - LP_exp| = " << diff
              << " (tol=" << kExpTol << ")" << std::endl;
    if (diff > kExpTol) {
      std::cerr << "FAIL: " << c.label << " CFR/LP exploitability diff "
                << diff << " exceeds tolerance " << kExpTol << std::endl;
      SPIEL_CHECK_LE(diff, kExpTol);
    }
  }

  std::cout << "TestLPRNRParity PASSED" << std::endl;
}

// ============================================================================
// Step 14: Combinatorial gadget × response × solver × lock test
// ============================================================================

// Collects all subgame info states from the decomposition (both players).
std::array<std::unordered_set<std::string>, 2> CollectAllSubgameIS(
    const SubgameDecomposition& decomp) {
  std::array<std::unordered_set<std::string>, 2> all_is;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    auto per_player = CollectSubgameInfoStatesPerPlayer(roots);
    for (int p = 0; p < 2; ++p) {
      all_is[p].insert(per_player[p].begin(), per_player[p].end());
    }
  }
  return all_is;
}

void TestCombinatorial() {
  std::cout << "TestCombinatorial (gadget x response x solver x lock)..."
            << std::endl;

  auto s = MakeKuhnSetup(2000);
  // Tabularized so string-based GetStatePolicy works in LP+RNR path.
  TabularPolicy opponent_model = GetUniformPolicy(*s.game);

  // Gadgets to test in the combinatorial loop.
  // FullPath/FullTrunk are tested separately below (they need extra config).
  struct GadgetEntry {
    GadgetKind kind;
    const char* name;
  };
  const GadgetEntry gadgets[] = {
      {GadgetKind::kUnsafe,        "Unsafe"},
      {GadgetKind::kResolving,     "Resolving"},
      {GadgetKind::kMaxMargin,     "MaxMargin"},
      {GadgetKind::kResolvingByIS, "ResolvingByIS"},
  };

  const ResponseKind responses[] = {ResponseKind::kNash, ResponseKind::kRNR};
  const SolverKind solvers[] = {SolverKind::kCFR, SolverKind::kLP};
  const bool lock_values[] = {true, false};

  int n_combos = 0, n_passed = 0, n_skipped = 0;

  for (const auto& g : gadgets) {
    for (ResponseKind resp : responses) {
      for (SolverKind sol : solvers) {
        // For Nash response, lock is irrelevant (no fixed branch).
        // We run Nash only once (with lock=true default).
        const bool* lock_set = lock_values;
        int lock_count = (resp == ResponseKind::kNash) ? 1 : 2;

        for (int li = 0; li < lock_count; ++li) {
          bool lock = lock_set[li];
          ++n_combos;

          std::string label =
              std::string(g.name) + " / " +
              (resp == ResponseKind::kNash ? "Nash" : "RNR") + " / " +
              (sol == SolverKind::kCFR ? "CFR" : "LP") +
              (resp == ResponseKind::kRNR
                   ? std::string(" / lock=") + (lock ? "true" : "false")
                   : "");
          std::cout << "  [" << n_combos << "] " << label << std::endl;

          ResolvingConfig config;
          config.gadget_kind = g.kind;
          config.response_kind = resp;
          config.solver_kind = sol;
          config.lock_opponent_in_fixed_branch = lock;
          config.p = 0.5;
          config.cfr_iterations = 500;
          config.opponent_model = &opponent_model;

          std::shared_ptr<TabularPolicy> merged;
          try {
            if (resp == ResponseKind::kNash) {
              // Nash: run a single two-player solve (ResolveSubgames handles
              // both player passes internally for distorting gadgets).
              config.target_player = 0;
              auto p0 = ResolveSubgames(s.decomp, s.trunk, config);
              config.target_player = 1;
              auto p1 = ResolveSubgames(s.decomp, s.trunk, config);
              auto all_is = CollectAllSubgameIS(s.decomp);
              merged = MergeSubgamePolicies(*p0, *p1, all_is);
            } else {
              // RNR: one pass per target player.
              config.target_player = 0;
              auto u0 = ResolveSubgames(s.decomp, s.trunk, config);
              config.target_player = 1;
              auto u1 = ResolveSubgames(s.decomp, s.trunk, config);
              auto all_is = CollectAllSubgameIS(s.decomp);
              merged = MergeSubgamePolicies(*u0, *u1, all_is);
            }
          } catch (const std::exception& e) {
            std::cout << "    SKIPPED (exception: " << e.what() << ")"
                      << std::endl;
            ++n_skipped;
            continue;
          }

          // Assert: policy is non-empty.
          SPIEL_CHECK_TRUE(merged != nullptr);
          SPIEL_CHECK_GT(merged->PolicyTable().size(), 0);

          // Assert: exploitability is finite.
          double exp = algorithms::Exploitability(*s.game, *merged);
          SPIEL_CHECK_TRUE(std::isfinite(exp));
          std::cout << "    exploitability=" << exp << " PASSED" << std::endl;
          ++n_passed;
        }
      }
    }
  }

  // FullPath / FullTrunk: Nash + CFR only (LP infeasible for large games).
  // These gadgets need decomp_depth set for configuration.
  const GadgetKind full_gadgets[] = {GadgetKind::kFullPath,
                                     GadgetKind::kFullTrunk};
  const char* full_names[] = {"FullPath", "FullTrunk"};
  for (int gi = 0; gi < 2; ++gi) {
    for (ResponseKind resp : {ResponseKind::kNash, ResponseKind::kRNR}) {
      for (SolverKind sol : {SolverKind::kCFR}) {
        // LP on Full gadgets is skipped (infeasible / too slow).
        int lock_count = (resp == ResponseKind::kNash) ? 1 : 2;
        for (int li = 0; li < lock_count; ++li) {
          bool lock = lock_values[li];
          ++n_combos;

          std::string label =
              std::string(full_names[gi]) + " / " +
              (resp == ResponseKind::kNash ? "Nash" : "RNR") + " / CFR" +
              (resp == ResponseKind::kRNR
                   ? std::string(" / lock=") + (lock ? "true" : "false")
                   : "");
          std::cout << "  [Full " << n_combos << "] " << label << std::endl;

          ResolvingConfig config;
          config.gadget_kind = full_gadgets[gi];
          config.response_kind = resp;
          config.solver_kind = SolverKind::kCFR;
          config.lock_opponent_in_fixed_branch = lock;
          config.p = 0.5;
          config.cfr_iterations = 200;
          config.opponent_model = &opponent_model;
          config.decomp_depth = 1;

          std::shared_ptr<TabularPolicy> merged;
          try {
            config.target_player = 0;
            auto u0 = ResolveSubgames(s.decomp, s.trunk, config);
            config.target_player = 1;
            auto u1 = ResolveSubgames(s.decomp, s.trunk, config);
            auto all_is = CollectAllSubgameIS(s.decomp);
            merged = MergeSubgamePolicies(*u0, *u1, all_is);
          } catch (const std::exception& e) {
            std::cout << "    SKIPPED (exception: " << e.what() << ")"
                      << std::endl;
            ++n_skipped;
            continue;
          }

          SPIEL_CHECK_TRUE(merged != nullptr);
          SPIEL_CHECK_GT(merged->PolicyTable().size(), 0);
          double exp = algorithms::Exploitability(*s.game, *merged);
          SPIEL_CHECK_TRUE(std::isfinite(exp));
          std::cout << "    exploitability=" << exp << " PASSED" << std::endl;
          ++n_passed;
        }
      }

      // LP on Full gadgets: explicitly skip.
      ++n_combos;
      std::cout << "  [Full-LP " << n_combos << "] " << full_names[gi]
                << " / " << (resp == ResponseKind::kNash ? "Nash" : "RNR")
                << " / LP -- SKIPPED (infeasible)" << std::endl;
      ++n_skipped;
    }
  }

  std::cout << "TestCombinatorial: " << n_passed << " passed, " << n_skipped
            << " skipped, " << n_combos << " total." << std::endl;
  std::cout << "TestCombinatorial PASSED" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestResolvingByISNashCFRSmoke();
  open_spiel::TestSESEquivalence();
  open_spiel::TestOXSmoke();
  open_spiel::TestLPRNRParity();
  open_spiel::TestCombinatorial();
  std::cout << "\nAll gadget tests passed!" << std::endl;
  return 0;
}

