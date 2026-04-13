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

#include "open_spiel/game_transforms/full_gadget.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ortools/linear_solver/linear_solver.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#include "open_spiel/algorithms/ortools/trunk_exploitability.h"
#include "open_spiel/game_transforms/matrix_valued_states.h"
#include "open_spiel/game_transforms/subgame_utils.h"
#include "open_spiel/game_transforms/unsafe_subgame.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

std::vector<double> ComputeReturnsUnderPolicy(const State& state,
                                              const Policy& policy) {
  if (state.IsTerminal()) return state.Returns();
  std::vector<double> ev(state.NumPlayers(), 0.0);
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      auto cev = ComputeReturnsUnderPolicy(*child, policy);
      for (int i = 0; i < ev.size(); ++i) ev[i] += p * cev[i];
    }
    return ev;
  }
  Player pl = state.CurrentPlayer();
  auto ap = policy.GetStatePolicy(state, pl);
  SPIEL_CHECK_FALSE(ap.empty());
  for (const auto& [a, p] : ap) {
    if (p <= 0.0) continue;
    auto child = state.Clone();
    child->ApplyAction(a);
    auto cev = ComputeReturnsUnderPolicy(*child, policy);
    for (int i = 0; i < ev.size(); ++i) ev[i] += p * cev[i];
  }
  return ev;
}

void StitchPlayerFromFGPolicy(
    const TabularPolicy& fg_policy,
    int resolving_player,
    const std::unordered_set<std::string>& allowed_infosets,
    const std::unordered_set<std::string>& trunk_infosets,
    TabularPolicy* combined) {
  const std::string prefix = "full_F:subgame:";
  const std::string expected_tag =
      absl::StrCat("full_F:subgame:P", resolving_player, ":");
  for (const auto& [sub_is, ap] : fg_policy.PolicyTable()) {
    if (sub_is.compare(0, prefix.size(), prefix) != 0) continue;
    if (sub_is.compare(0, expected_tag.size(), expected_tag) != 0) continue;
    std::string orig = sub_is.substr(prefix.size());
    if (orig.size() > 3 && orig[0] == 'P' &&
        (orig[1] == '0' || orig[1] == '1') && orig[2] == ':') {
      orig = orig.substr(3);
    }
    if (allowed_infosets.count(orig) == 0) continue;
    if (trunk_infosets.count(orig) > 0) {
      SpielFatalError(absl::StrCat(
          "FG stitching attempted to overwrite trunk infoset: ", orig,
          " for resolving_player=", resolving_player));
    }
    ActionsAndProbs clamped;
    double sum = 0.0;
    for (const auto& [a, p] : ap) {
      double cp = std::max(0.0, p);
      clamped.push_back({a, cp});
      sum += cp;
    }
    SPIEL_CHECK_GT(sum, 0.0);
    for (auto& [a, p] : clamped) p /= sum;
    combined->SetStatePolicy(orig, clamped);
  }
}

TabularPolicy SolveHalfTargetForPlayer(
    std::shared_ptr<const Game> game,
    const SubgameDecomposition& decomp,
    const std::vector<std::string>& target_groups,
    const std::vector<std::string>& other_groups,
    const TabularPolicy& trunk_policy,
    const std::unordered_set<std::string>& trunk_infosets,
    const Policy* other_half_reference_policy,
    int resolving_player) {
  auto trunk_ptr = std::make_shared<TabularPolicy>(trunk_policy);
  TabularPolicy solved_slice;
  std::unordered_set<std::string> target_set(target_groups.begin(),
                                             target_groups.end());
  std::unordered_set<std::string> other_set(other_groups.begin(),
                                            other_groups.end());

  // Build exactly two boundary groups for this solve: target half and other half.
  const std::string kTargetHalf = "__half_target__";
  const std::string kOtherHalf = "__half_other__";
  std::unordered_map<std::string, std::vector<std::string>> boundary_by_group;
  std::unordered_map<std::string, std::vector<double>> boundary_values;
  std::unordered_set<std::string> allowed_infosets;
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    const bool in_target = target_set.count(pub_obs) > 0;
    const bool in_other = other_set.count(pub_obs) > 0;
    if (!in_target && !in_other) continue;
    const std::string bucket = in_target ? kTargetHalf : kOtherHalf;
    for (const auto& root : roots) {
      boundary_by_group[bucket].push_back(root->HistoryString());
      if (in_other && other_half_reference_policy != nullptr) {
        boundary_values[root->HistoryString()] =
            ComputeReturnsUnderPolicy(*root, *other_half_reference_policy);
      } else {
        boundary_values[root->HistoryString()] = {0.0, 0.0};  // MVS boundaries
      }
    }
    if (in_target) {
      auto info = CollectSubgameInfoStatesPerPlayer(roots);
      allowed_infosets.insert(info[resolving_player].begin(),
                              info[resolving_player].end());
    }
  }

  auto fg = CreateFullGadgetGame(
      game, trunk_ptr, resolving_player, kTargetHalf,
      boundary_by_group, FullGadgetGame::Mode::kTrunk,
      /*boundary_portfolios_p0=*/{},
      /*boundary_portfolios_p1=*/{},
      /*enumerate_boundary_portfolios=*/other_half_reference_policy == nullptr);
  algorithms::ortools::SequenceFormLpSpecification spec(
      *fg, "CLP", /*return_nan_if_non_optimal=*/false);
  spec.SpecifyLinearProgram(resolving_player);
  double val = spec.Solve();
  TabularPolicy fg_policy = spec.OptimalPolicy(
      resolving_player, /*uniform_imputation=*/false);
  TabularPolicy fg_policy_imputed = spec.OptimalPolicy(
      resolving_player, /*uniform_imputation=*/true);
  TabularPolicy solved_slice_imputed;
  std::cout << "    p=" << resolving_player
            << " half_target_size=" << target_groups.size()
            << " val=" << val << std::endl;
  StitchPlayerFromFGPolicy(fg_policy, resolving_player, allowed_infosets,
                           trunk_infosets,
                           &solved_slice);
  StitchPlayerFromFGPolicy(fg_policy_imputed, resolving_player, allowed_infosets,
                           trunk_infosets,
                           &solved_slice_imputed);
  int sparse_rows = 0;
  int dense_rows = 0;
  int dense_only = 0;
  std::vector<std::string> dense_only_examples;
  for (const auto& is : allowed_infosets) {
    bool has_sparse = !solved_slice.GetStatePolicy(is).empty();
    bool has_dense = !solved_slice_imputed.GetStatePolicy(is).empty();
    if (has_sparse) ++sparse_rows;
    if (has_dense) ++dense_rows;
    if (!has_sparse && has_dense) {
      ++dense_only;
      if (dense_only_examples.size() < 4) dense_only_examples.push_back(is);
    }
  }
  std::cout << "      slice rows (sparse/dense)=" << sparse_rows << "/"
            << dense_rows << " dense_only=" << dense_only << std::endl;
  for (const auto& ex : dense_only_examples) {
    std::cout << "        dense-only example: " << ex << std::endl;
  }
  return solved_slice;
}

std::vector<std::string> SortedGroups(const SubgameDecomposition& decomp) {
  std::vector<std::string> groups;
  groups.reserve(decomp.grouped_subgames.size());
  for (const auto& [k, _] : decomp.grouped_subgames) groups.push_back(k);
  std::sort(groups.begin(), groups.end());
  return groups;
}

std::unordered_set<std::string> CollectPlayerSubgameInfosetsForGroups(
    const SubgameDecomposition& decomp,
    const std::vector<std::string>& groups,
    int player) {
  std::unordered_set<std::string> out;
  std::unordered_set<std::string> selected(groups.begin(), groups.end());
  for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
    if (selected.count(pub_obs) == 0) continue;
    auto info = CollectSubgameInfoStatesPerPlayer(roots);
    out.insert(info[player].begin(), info[player].end());
  }
  return out;
}

void PrintHalfCoverageDiagnostics(
    int player,
    const std::unordered_set<std::string>& expected_a,
    const std::unordered_set<std::string>& expected_b,
    const TabularPolicy& half_a,
    const TabularPolicy& half_b) {
  int got_a = 0, got_b = 0, got_union = 0, got_overlap = 0;
  int missing_a = 0, missing_b = 0;
  std::unordered_set<std::string> union_expected = expected_a;
  union_expected.insert(expected_b.begin(), expected_b.end());
  for (const auto& is : expected_a) {
    if (!half_a.GetStatePolicy(is).empty()) ++got_a;
    else ++missing_a;
  }
  for (const auto& is : expected_b) {
    if (!half_b.GetStatePolicy(is).empty()) ++got_b;
    else ++missing_b;
  }
  for (const auto& is : union_expected) {
    bool in_a = !half_a.GetStatePolicy(is).empty();
    bool in_b = !half_b.GetStatePolicy(is).empty();
    if (in_a || in_b) ++got_union;
    if (in_a && in_b) ++got_overlap;
  }
  std::cout << "  Coverage P" << player
            << " expected(A/B/union)=" << expected_a.size() << "/"
            << expected_b.size() << "/" << union_expected.size()
            << " got(A/B/union/overlap)=" << got_a << "/" << got_b << "/"
            << got_union << "/" << got_overlap
            << " missing(A/B)=" << missing_a << "/" << missing_b
            << std::endl;
}

double L1PolicyDelta(const ActionsAndProbs& a, const ActionsAndProbs& b) {
  std::unordered_map<Action, double> pa;
  std::unordered_map<Action, double> pb;
  for (const auto& [act, p] : a) pa[act] = p;
  for (const auto& [act, p] : b) pb[act] = p;
  double l1 = 0.0;
  for (const auto& [act, p] : pa) {
    double q = (pb.count(act) > 0) ? pb[act] : 0.0;
    l1 += std::abs(p - q);
  }
  for (const auto& [act, q] : pb) {
    if (pa.count(act) == 0) l1 += std::abs(q);
  }
  return l1;
}

// Clamp negative LP noise and renormalize so TabularBestResponse never sees p<0.
void NormalizeNonnegativePolicy(
    TabularPolicy* policy,
    const std::array<std::unordered_set<std::string>, 2>& all_infosets) {
  for (int pl = 0; pl < 2; ++pl) {
    for (const auto& is : all_infosets[pl]) {
      ActionsAndProbs ap = policy->GetStatePolicy(is);
      if (ap.empty()) continue;
      ActionsAndProbs out;
      double sum = 0.0;
      for (const auto& [a, p] : ap) {
        double q = std::max(0.0, p);
        out.push_back({a, q});
        sum += q;
      }
      if (sum <= 1e-30) continue;
      for (auto& [a, p] : out) p /= sum;
      policy->SetStatePolicy(is, out);
    }
  }
}

std::array<std::unordered_set<std::string>, 2> CollectAllInfoStates(
    const Game& game) {
  std::array<std::unordered_set<std::string>, 2> result;
  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
      return;
    }
    Player pl = state.CurrentPlayer();
    if (pl >= 0 && pl < 2) {
      result[pl].insert(state.InformationStateString(pl));
    }
    for (Action a : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      traverse(*child);
    }
  };
  traverse(*game.NewInitialState());
  return result;
}

void FillMissingWithUniform(const Game& game, TabularPolicy* policy,
                            const std::unordered_set<std::string>& need_p0,
                            const std::unordered_set<std::string>& need_p1) {
  std::function<void(const State&)> traverse = [&](const State& state) {
    if (state.IsTerminal()) return;
    if (state.IsChanceNode()) {
      for (const auto& [a, p] : state.ChanceOutcomes()) {
        auto child = state.Clone();
        child->ApplyAction(a);
        traverse(*child);
      }
      return;
    }
    Player pl = state.CurrentPlayer();
    if (pl == 0 || pl == 1) {
      const std::string is = state.InformationStateString(pl);
      const bool needed =
          (pl == 0) ? (need_p0.count(is) > 0) : (need_p1.count(is) > 0);
      if (needed && policy->GetStatePolicy(is).empty()) {
        auto legal = state.LegalActions();
        SPIEL_CHECK_FALSE(legal.empty());
        const double p = 1.0 / static_cast<double>(legal.size());
        ActionsAndProbs ap;
        ap.reserve(legal.size());
        for (Action a : legal) ap.push_back({a, p});
        policy->SetStatePolicy(is, ap);
      }
    }
    for (Action a : state.LegalActions()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      traverse(*child);
    }
  };
  traverse(*game.NewInitialState());
}

void FillMissingFromReference(TabularPolicy* policy, const TabularPolicy& ref,
                              const std::unordered_set<std::string>& need_p0,
                              const std::unordered_set<std::string>& need_p1) {
  for (const auto& is : need_p0) {
    if (policy->GetStatePolicy(is).empty()) {
      auto ap = ref.GetStatePolicy(is);
      SPIEL_CHECK_FALSE(ap.empty());
      policy->SetStatePolicy(is, ap);
    }
  }
  for (const auto& is : need_p1) {
    if (policy->GetStatePolicy(is).empty()) {
      auto ap = ref.GetStatePolicy(is);
      SPIEL_CHECK_FALSE(ap.empty());
      policy->SetStatePolicy(is, ap);
    }
  }
}

void PrintTopDiffs(const TabularPolicy& ref, const TabularPolicy& cand,
                   const std::unordered_set<std::string>& infosets,
                   int player, int top_k = 12) {
  std::vector<std::pair<double, std::string>> diffs;
  diffs.reserve(infosets.size());
  int nonzero = 0;
  double total = 0.0;
  for (const auto& is : infosets) {
    auto a = ref.GetStatePolicy(is);
    auto b = cand.GetStatePolicy(is);
    if (a.empty() || b.empty()) continue;
    double d = L1PolicyDelta(a, b);
    total += d;
    if (d > 1e-10) {
      ++nonzero;
      diffs.push_back({d, is});
    }
  }
  std::sort(diffs.begin(), diffs.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });
  std::cout << "  P" << player << " differing infosets: " << nonzero
            << ", total L1 delta: " << total << std::endl;
  for (int i = 0; i < std::min(top_k, static_cast<int>(diffs.size())); ++i) {
    const auto& [d, is] = diffs[i];
    std::cout << "    [P" << player << " top-" << (i + 1) << "] L1=" << d
              << " is=" << is << std::endl;
  }
}

std::string APToString(const ActionsAndProbs& ap) {
  std::string out = "[";
  bool first = true;
  for (const auto& [a, p] : ap) {
    if (!first) out += ", ";
    first = false;
    out += absl::StrCat(a, ":", p);
  }
  out += "]";
  return out;
}

std::string ExtractPublicTag(const std::string& info_state) {
  const std::string marker = "[Public: ";
  size_t p = info_state.find(marker);
  if (p == std::string::npos) return "NA";
  p += marker.size();
  size_t q = info_state.find("]", p);
  if (q == std::string::npos || q <= p) return "NA";
  return info_state.substr(p, q - p);
}

int CountTerminalStates(const State& state) {
  if (state.IsTerminal()) return 1;
  int total = 0;
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      total += CountTerminalStates(*child);
    }
    return total;
  }
  for (Action a : state.LegalActions()) {
    auto child = state.Clone();
    child->ApplyAction(a);
    total += CountTerminalStates(*child);
  }
  return total;
}

void CollectTerminalPayoffsP1(const State& state, std::vector<double>* out) {
  if (state.IsTerminal()) {
    out->push_back(state.Returns()[1]);
    return;
  }
  if (state.IsChanceNode()) {
    for (const auto& [a, p] : state.ChanceOutcomes()) {
      auto child = state.Clone();
      child->ApplyAction(a);
      CollectTerminalPayoffsP1(*child, out);
    }
    return;
  }
  for (Action a : state.LegalActions()) {
    auto child = state.Clone();
    child->ApplyAction(a);
    CollectTerminalPayoffsP1(*child, out);
  }
}

std::vector<double> UniqueSortedTol(std::vector<double> vals, double tol = 1e-9) {
  std::sort(vals.begin(), vals.end());
  std::vector<double> out;
  for (double v : vals) {
    if (out.empty() || std::abs(v - out.back()) > tol) out.push_back(v);
  }
  return out;
}

bool SetEqualTol(const std::vector<double>& a,
                 const std::vector<double>& b,
                 double tol = 1e-8) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

std::string VecToString(const std::vector<double>& v) {
  std::string s = "{";
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (i > 0) s += ", ";
    s += absl::StrCat(v[i]);
  }
  s += "}";
  return s;
}

void PrintRootTreeAndMatrix(const State& root) {
  std::cout << "\n=== One-root tree + matrix view ===" << std::endl;
  std::cout << "root history: " << root.HistoryString() << std::endl;
  std::cout << "root P0 infoset: " << root.InformationStateString(0) << std::endl;
  std::cout << "root P1 infoset: " << root.InformationStateString(1) << std::endl;

  // Collect infosets + legal actions reachable in this subtree.
  std::array<std::unordered_map<std::string, std::vector<Action>>, 2> legal_by_is;
  std::function<void(const State&)> collect = [&](const State& s) {
    if (s.IsTerminal()) return;
    if (s.IsChanceNode()) {
      for (const auto& [a, p] : s.ChanceOutcomes()) {
        auto c = s.Clone();
        c->ApplyAction(a);
        collect(*c);
      }
      return;
    }
    Player pl = s.CurrentPlayer();
    if (pl == 0 || pl == 1) {
      std::string is = s.InformationStateString(pl);
      if (legal_by_is[pl].count(is) == 0) {
        legal_by_is[pl][is] = s.LegalActions();
      }
    }
    for (Action a : s.LegalActions()) {
      auto c = s.Clone();
      c->ApplyAction(a);
      collect(*c);
    }
  };
  collect(root);

  auto print_tree = [&](const State& s, const auto& self, int depth) -> void {
    std::string indent(depth * 2, ' ');
    if (s.IsTerminal()) {
      auto r = s.Returns();
      std::cout << indent << "T returns=(" << r[0] << ", " << r[1] << ")"
                << " hist=" << s.HistoryString() << std::endl;
      return;
    }
    if (s.IsChanceNode()) {
      std::cout << indent << "C chance outcomes:";
      for (const auto& [a, p] : s.ChanceOutcomes()) {
        std::cout << " " << a << "@" << p;
      }
      std::cout << std::endl;
      for (const auto& [a, p] : s.ChanceOutcomes()) {
        auto c = s.Clone();
        c->ApplyAction(a);
        std::cout << indent << "  -> a=" << a << std::endl;
        self(*c, self, depth + 1);
      }
      return;
    }
    Player pl = s.CurrentPlayer();
    std::cout << indent << "P" << pl << " is=" << s.InformationStateString(pl)
              << " legal={";
    bool first = true;
    for (Action a : s.LegalActions()) {
      if (!first) std::cout << ",";
      first = false;
      std::cout << a;
    }
    std::cout << "}" << std::endl;
    for (Action a : s.LegalActions()) {
      auto c = s.Clone();
      c->ApplyAction(a);
      std::cout << indent << "  -> a=" << a << std::endl;
      self(*c, self, depth + 1);
    }
  };
  print_tree(root, print_tree, 0);

  // Build pure strategy portfolios and print matrix.
  auto port_p0 = EnumerateSubtreePureStrategies(root, 0);
  auto port_p1 = EnumerateSubtreePureStrategies(root, 1);
  auto port_p0_red = EnumerateSubtreePureStrategiesReduced(root, 0);
  auto port_p1_red = EnumerateSubtreePureStrategiesReduced(root, 1);
  std::cout << "infosets: P0=" << legal_by_is[0].size()
            << " P1=" << legal_by_is[1].size() << std::endl;
  std::cout << "portfolio sizes: K0=" << port_p0.size()
            << " K1=" << port_p1.size() << std::endl;
  std::cout << "reduced portfolio sizes: K0=" << port_p0_red.size()
            << " K1=" << port_p1_red.size() << std::endl;

  std::vector<std::string> is0, is1;
  is0.reserve(legal_by_is[0].size());
  is1.reserve(legal_by_is[1].size());
  for (const auto& [k, _] : legal_by_is[0]) is0.push_back(k);
  for (const auto& [k, _] : legal_by_is[1]) is1.push_back(k);
  std::sort(is0.begin(), is0.end());
  std::sort(is1.begin(), is1.end());

  auto chosen_action_at = [](const Policy& pol, const std::string& is) {
    auto ap = pol.GetStatePolicy(is);
    Action chosen = kInvalidAction;
    for (const auto& [a, p] : ap) {
      if (p > 0.5) {
        chosen = a;
        break;
      }
    }
    return chosen;
  };

  std::cout << "\nP0 infoset order (row tuple coordinates):" << std::endl;
  for (int k = 0; k < static_cast<int>(is0.size()); ++k) {
    std::cout << "  i0[" << k << "] = " << is0[k] << std::endl;
  }
  std::cout << "P1 infoset order (column tuple coordinates):" << std::endl;
  for (int k = 0; k < static_cast<int>(is1.size()); ++k) {
    std::cout << "  i1[" << k << "] = " << is1[k] << std::endl;
  }

  auto tuple_only = [&](const Policy& pol, const std::vector<std::string>& infosets) {
    std::string out = "{";
    for (int idx = 0; idx < static_cast<int>(infosets.size()); ++idx) {
      if (idx > 0) out += ", ";
      out += absl::StrCat(chosen_action_at(pol, infosets[idx]));
    }
    out += "}";
    return out;
  };

  std::cout << "\nRow strategies (P0):" << std::endl;
  for (int i = 0; i < static_cast<int>(port_p0.size()); ++i) {
    std::cout << "  r" << i << " = " << tuple_only(*port_p0[i], is0) << std::endl;
    for (int k = 0; k < static_cast<int>(is0.size()); ++k) {
      std::cout << "    i0[" << k << "] -> a="
                << chosen_action_at(*port_p0[i], is0[k]) << std::endl;
    }
  }
  std::cout << "Column strategies (P1):" << std::endl;
  for (int j = 0; j < static_cast<int>(port_p1.size()); ++j) {
    std::cout << "  c" << j << " = " << tuple_only(*port_p1[j], is1) << std::endl;
    for (int k = 0; k < static_cast<int>(is1.size()); ++k) {
      std::cout << "    i1[" << k << "] -> a="
                << chosen_action_at(*port_p1[j], is1[k]) << std::endl;
    }
  }

  std::cout << "\nMatrix (P1 payoff):" << std::endl;
  for (int i = 0; i < static_cast<int>(port_p0.size()); ++i) {
    std::cout << "r" << i << ": ";
    for (int j = 0; j < static_cast<int>(port_p1.size()); ++j) {
      std::vector<const Policy*> pols = {port_p0[i].get(), port_p1[j].get()};
      auto ret = algorithms::ExpectedReturns(
          root, pols, /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
      std::cout << ret[1];
      if (j + 1 < static_cast<int>(port_p1.size())) std::cout << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nReduced matrix (P1 payoff):" << std::endl;
  for (int i = 0; i < static_cast<int>(port_p0_red.size()); ++i) {
    std::cout << "r" << i << ": ";
    for (int j = 0; j < static_cast<int>(port_p1_red.size()); ++j) {
      std::vector<const Policy*> pols = {port_p0_red[i].get(),
                                         port_p1_red[j].get()};
      auto ret = algorithms::ExpectedReturns(
          root, pols, /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
      std::cout << ret[1];
      if (j + 1 < static_cast<int>(port_p1_red.size())) std::cout << " ";
    }
    std::cout << std::endl;
  }
}

void PrintHalfComparisonForPlayer(
    const TabularPolicy& full_ne, const TabularPolicy& stitched,
    const TabularPolicy& half_a, const TabularPolicy& half_b,
    const std::unordered_set<std::string>& infosets, int player, int top_k = 12) {
  std::vector<std::pair<double, std::string>> diffs;
  for (const auto& is : infosets) {
    auto ne = full_ne.GetStatePolicy(is);
    auto st = stitched.GetStatePolicy(is);
    if (ne.empty() || st.empty()) continue;
    double d = L1PolicyDelta(ne, st);
    if (d > 1e-10) diffs.push_back({d, is});
  }
  std::sort(diffs.begin(), diffs.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });
  std::cout << "  --- Detailed half comparison for P" << player << " ---"
            << std::endl;
  for (int i = 0; i < std::min(top_k, static_cast<int>(diffs.size())); ++i) {
    const auto& [d, is] = diffs[i];
    auto ne = full_ne.GetStatePolicy(is);
    auto st = stitched.GetStatePolicy(is);
    auto a = half_a.GetStatePolicy(is);
    auto b = half_b.GetStatePolicy(is);
    std::cout << "    is=" << is << "\n"
              << "      L1(stitched,NE)=" << d << "\n"
              << "      NE:      " << APToString(ne) << "\n"
              << "      half-A:  " << APToString(a) << "\n"
              << "      half-B:  " << APToString(b) << "\n"
              << "      stitched:" << APToString(st) << std::endl;
  }
}

void PrintReachWeightedDiffs(
    const Game& game, const TabularPolicy& ref, const TabularPolicy& cand,
    int player, int top_k = 20) {
  std::unordered_map<std::string, double> reach;
  std::function<void(const State&, double)> traverse =
      [&](const State& state, double prob) {
        if (state.IsTerminal()) return;
        if (state.IsChanceNode()) {
          for (const auto& [a, p] : state.ChanceOutcomes()) {
            if (p <= 0.0) continue;
            auto child = state.Clone();
            child->ApplyAction(a);
            traverse(*child, prob * p);
          }
          return;
        }
        Player pl = state.CurrentPlayer();
        if (pl == 0 || pl == 1) {
          const std::string is = state.InformationStateString(pl);
          if (pl == player) reach[is] += prob;
          auto ap = cand.GetStatePolicy(is);
          if (ap.empty()) return;
          for (const auto& [a, p] : ap) {
            if (p <= 0.0) continue;
            auto child = state.Clone();
            child->ApplyAction(a);
            traverse(*child, prob * p);
          }
          return;
        }
      };
  traverse(*game.NewInitialState(), 1.0);

  std::vector<std::tuple<double, double, std::string>> rows;
  double total_weighted_l1 = 0.0;
  int positive_reach_infosets = 0;
  for (const auto& [is, r] : reach) {
    if (r <= 0.0) continue;
    auto a = ref.GetStatePolicy(is);
    auto b = cand.GetStatePolicy(is);
    if (a.empty() || b.empty()) continue;
    double l1 = L1PolicyDelta(a, b);
    double w = r * l1;
    total_weighted_l1 += w;
    ++positive_reach_infosets;
    if (l1 > 1e-12) rows.push_back({w, l1, is});
  }
  std::sort(rows.begin(), rows.end(),
            [](const auto& x, const auto& y) { return std::get<0>(x) > std::get<0>(y); });
  std::cout << "  Reach-weighted diffs P" << player
            << ": positive_reach_infosets=" << positive_reach_infosets
            << " weighted_L1_total=" << total_weighted_l1 << std::endl;
  for (int i = 0; i < std::min(top_k, static_cast<int>(rows.size())); ++i) {
    const auto& [w, l1, is] = rows[i];
    std::cout << "    [P" << player << " reach-top-" << (i + 1)
              << "] weighted=" << w << " l1=" << l1
              << " reach=" << reach[is] << " is=" << is << std::endl;
  }
}

void RunLeducHalfSplitExperiment() {
  std::cout << "RunLeducHalfSplitExperiment..." << std::endl;
  auto game = LoadGame("leduc_poker");
  TabularPolicy uniform_base(*game);
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto groups = SortedGroups(decomp);
  SPIEL_CHECK_FALSE(groups.empty());

  auto mvs = CreateMVSGameWithSubtreePureStrategies(
      game, 3, MVSGame::DepthMode::kRoundBased);
  auto [mvs_ne, mvs_val] = algorithms::ortools::MakeEquilibriumPolicy(*mvs, true);
  double mvs_exp = algorithms::Exploitability(*mvs, mvs_ne);
  std::cout << "  MVS LP exp (in MVS game): " << mvs_exp << std::endl;

  auto [full_ne, full_val] = algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  double full_ne_exp = algorithms::Exploitability(*game, full_ne);
  std::cout << "  Full LP NE exp: " << full_ne_exp << std::endl;

  auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::unordered_set<std::string> trunk_all_infosets;
  TabularPolicy mvs_trunk;
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& is : info_states) {
      trunk_all_infosets.insert(is);
      ActionsAndProbs ap = mvs_ne.GetStatePolicy(is);
      if (ap.empty()) {
        SpielFatalError(absl::StrCat(
            "Missing MVS trunk policy for infoset: ", is, " player=", player));
      }
      mvs_trunk.SetStatePolicy(is, ap);
    }
  }

  const int n = groups.size();
  const int half = n / 2;
  std::vector<std::string> first_half(groups.begin(), groups.begin() + half);
  std::vector<std::string> second_half(groups.begin() + half, groups.end());
  std::cout << "  Groups total/half: " << n << "/" << half << std::endl;

  auto run_half_split_with_trunk = [&](const TabularPolicy& trunk_policy,
                                       const Policy* other_half_ref_policy,
                                       const std::string& label) {
    std::cout << "  Two-solve-per-player half pass [" << label << "]..."
              << std::endl;
    TabularPolicy stitched_split = trunk_policy;
    TabularPolicy p0_half_a =
        SolveHalfTargetForPlayer(game, decomp, first_half, second_half, trunk_policy,
                                 trunk_all_infosets, other_half_ref_policy, 0);
    TabularPolicy p0_half_b =
        SolveHalfTargetForPlayer(game, decomp, second_half, first_half, trunk_policy,
                                 trunk_all_infosets, other_half_ref_policy, 0);
    TabularPolicy p1_half_a =
        SolveHalfTargetForPlayer(game, decomp, first_half, second_half, trunk_policy,
                                 trunk_all_infosets, other_half_ref_policy, 1);
    TabularPolicy p1_half_b =
        SolveHalfTargetForPlayer(game, decomp, second_half, first_half, trunk_policy,
                                 trunk_all_infosets, other_half_ref_policy, 1);
    auto p0_expected_a =
        CollectPlayerSubgameInfosetsForGroups(decomp, first_half, 0);
    auto p0_expected_b =
        CollectPlayerSubgameInfosetsForGroups(decomp, second_half, 0);
    auto p1_expected_a =
        CollectPlayerSubgameInfosetsForGroups(decomp, first_half, 1);
    auto p1_expected_b =
        CollectPlayerSubgameInfosetsForGroups(decomp, second_half, 1);
    PrintHalfCoverageDiagnostics(0, p0_expected_a, p0_expected_b, p0_half_a, p0_half_b);
    PrintHalfCoverageDiagnostics(1, p1_expected_a, p1_expected_b, p1_half_a, p1_half_b);
    for (const auto& [is, ap] : p0_half_a.PolicyTable())
      stitched_split.SetStatePolicy(is, ap);
    for (const auto& [is, ap] : p0_half_b.PolicyTable())
      stitched_split.SetStatePolicy(is, ap);
    for (const auto& [is, ap] : p1_half_a.PolicyTable())
      stitched_split.SetStatePolicy(is, ap);
    for (const auto& [is, ap] : p1_half_b.PolicyTable())
      stitched_split.SetStatePolicy(is, ap);

    std::unordered_set<std::string> p0_expected_all = p0_expected_a;
    p0_expected_all.insert(p0_expected_b.begin(), p0_expected_b.end());
    std::unordered_set<std::string> p1_expected_all = p1_expected_a;
    p1_expected_all.insert(p1_expected_b.begin(), p1_expected_b.end());
    int missing_p0 = 0, missing_p1 = 0;
    for (const auto& is : p0_expected_all) {
      if (stitched_split.GetStatePolicy(is).empty()) ++missing_p0;
    }
    for (const auto& is : p1_expected_all) {
      if (stitched_split.GetStatePolicy(is).empty()) ++missing_p1;
    }
    std::cout << "  [" << label << "] Missing stitched rows P0/P1: " << missing_p0
              << "/" << missing_p1 << std::endl;

    TabularPolicy stitched_uniform = stitched_split;
    FillMissingWithUniform(*game, &stitched_uniform, p0_expected_all, p1_expected_all);
    double exp_uniform = algorithms::Exploitability(*game, stitched_uniform);
    std::cout << "  [" << label << "] Exploitability (missing->uniform): "
              << exp_uniform << std::endl;

    TabularPolicy stitched_ref = stitched_split;
    FillMissingFromReference(&stitched_ref, full_ne, p0_expected_all, p1_expected_all);
    double exp_ref = algorithms::Exploitability(*game, stitched_ref);
    std::cout << "  [" << label << "] Exploitability (missing->global_NE): "
              << exp_ref << std::endl;

    auto all_infosets = CollectAllInfoStates(*game);
    std::cout << "  --- Diff vs full LP NE [" << label << "] ---" << std::endl;
    PrintTopDiffs(full_ne, stitched_ref, all_infosets[0], 0);
    PrintTopDiffs(full_ne, stitched_ref, all_infosets[1], 1);
    PrintReachWeightedDiffs(*game, full_ne, stitched_ref, 0, 12);
    PrintReachWeightedDiffs(*game, full_ne, stitched_ref, 1, 12);
  };

  run_half_split_with_trunk(mvs_trunk, /*other_half_ref_policy=*/nullptr,
                            "MVS trunk + MVS other-half");
  run_half_split_with_trunk(full_ne, /*other_half_ref_policy=*/nullptr,
                            "Full-NE trunk + MVS other-half");

  // Scalar boundary-value controls were removed from FullGadget. Keep only
  // portfolio-based boundary runs.

  // --- Diagnostic: Joint-solve stitching test ---
  std::cout << "\n=== Joint-Solve Stitching Diagnostic ===" << std::endl;
  {
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    auto all_is_orig = CollectAllInfoStates(*game);

    // Helper: create FullGadget game with given target/other groups
    auto make_fg = [&](int rp,
                       const std::vector<std::string>& target_groups,
                       const std::vector<std::string>& other_groups) {
      std::unordered_map<std::string, std::vector<std::string>> bbg;
      std::unordered_map<std::string, std::vector<double>> bv;
      const std::string kT = "__jtarget__";
      const std::string kO = "__jother__";
      std::unordered_set<std::string> ts(target_groups.begin(),
                                         target_groups.end());
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        const std::string bucket = ts.count(pub_obs) ? kT : kO;
        for (const auto& root : roots) {
          bbg[bucket].push_back(root->HistoryString());
          bv[root->HistoryString()] = {0.0, 0.0};
        }
      }
      return CreateFullGadgetGame(
          game, trunk_ptr, rp, kT, bbg, FullGadgetGame::Mode::kTrunk,
          {}, {}, /*enumerate_boundary_portfolios=*/true);
    };

    // Helper: stitch both players' target-half strategies from joint policy
    auto stitch_both_players = [&](const TabularPolicy& fg_policy,
                                   const std::vector<std::string>& target_groups,
                                   TabularPolicy* out) {
      for (int pl = 0; pl < 2; ++pl) {
        std::unordered_set<std::string> allowed;
        for (const auto& g : target_groups) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            allowed.insert(info[pl].begin(), info[pl].end());
          }
        }
        StitchPlayerFromFGPolicy(fg_policy, pl, allowed,
                                 trunk_all_infosets, out);
      }
    };

    // Test A: single rp, stitch both players from joint solve of one half
    for (int rp = 0; rp < 1; ++rp) {  // only rp=0 for now (rp=1 LP crashes)
      std::cout << "  [Test A] rp=" << rp
                << " joint-solve, stitch both players for first-half" << std::endl;
      auto fg_a = make_fg(rp, first_half, second_half);
      auto [ne_a, val_a] =
          algorithms::ortools::MakeEquilibriumPolicy(*fg_a, true);
      std::cout << "    val=" << val_a << std::endl;

      TabularPolicy stitch_a = full_ne;
      stitch_both_players(ne_a, first_half, &stitch_a);
      FillMissingFromReference(&stitch_a, full_ne,
                               all_is_orig[0], all_is_orig[1]);
      double exp_a = algorithms::Exploitability(*game, stitch_a);
      std::cout << "    first-half only exp=" << exp_a << std::endl;

      // Test B: joint-solve both halves with SAME rp, stitch together
      std::cout << "  [Test B] rp=" << rp
                << " joint-solve BOTH halves, same rp" << std::endl;
      auto fg_b = make_fg(rp, second_half, first_half);
      auto [ne_b, val_b] =
          algorithms::ortools::MakeEquilibriumPolicy(*fg_b, true);
      std::cout << "    half-A val=" << val_a << " half-B val=" << val_b
                << std::endl;

      TabularPolicy stitch_b = full_ne;
      stitch_both_players(ne_a, first_half, &stitch_b);
      stitch_both_players(ne_b, second_half, &stitch_b);
      FillMissingFromReference(&stitch_b, full_ne,
                               all_is_orig[0], all_is_orig[1]);
      double exp_b = algorithms::Exploitability(*game, stitch_b);
      std::cout << "    both-halves exp=" << exp_b << std::endl;

      PrintTopDiffs(full_ne, stitch_b, all_is_orig[0], 0, 6);
      PrintTopDiffs(full_ne, stitch_b, all_is_orig[1], 1, 6);

      // Test C: per-player solves but from SAME rp game
      std::cout << "  [Test C] rp=" << rp
                << " per-player solve from same rp, stitch both halves"
                << std::endl;
      algorithms::ortools::SequenceFormLpSpecification spec_a(
          *fg_a, "CLP", false);
      spec_a.SpecifyLinearProgram(rp);
      spec_a.Solve();
      TabularPolicy pp_a = spec_a.OptimalPolicy(rp, true);

      algorithms::ortools::SequenceFormLpSpecification spec_b(
          *fg_b, "CLP", false);
      spec_b.SpecifyLinearProgram(rp);
      spec_b.Solve();
      TabularPolicy pp_b = spec_b.OptimalPolicy(rp, true);

      TabularPolicy stitch_c = full_ne;
      for (int pl = 0; pl < 2; ++pl) {
        if (pl == rp) {
          std::unordered_set<std::string> allowed_a;
          for (const auto& g : first_half) {
            auto it2 = decomp.grouped_subgames.find(g);
            if (it2 != decomp.grouped_subgames.end()) {
              auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
              allowed_a.insert(info[pl].begin(), info[pl].end());
            }
          }
          StitchPlayerFromFGPolicy(pp_a, rp, allowed_a,
                                   trunk_all_infosets, &stitch_c);

          std::unordered_set<std::string> allowed_b;
          for (const auto& g : second_half) {
            auto it2 = decomp.grouped_subgames.find(g);
            if (it2 != decomp.grouped_subgames.end()) {
              auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
              allowed_b.insert(info[pl].begin(), info[pl].end());
            }
          }
          StitchPlayerFromFGPolicy(pp_b, rp, allowed_b,
                                   trunk_all_infosets, &stitch_c);
        }
      }
      FillMissingFromReference(&stitch_c, full_ne,
                               all_is_orig[0], all_is_orig[1]);
      double exp_c = algorithms::Exploitability(*game, stitch_c);
      std::cout << "    per-player same-rp exp=" << exp_c << std::endl;
    }

    // Test D: P1 only from rp=1, keep P0 as NE
    std::cout << "  [Test D] P1 from rp=1 per-player, keep P0 as NE"
              << std::endl;
    {
      auto fg_d1 = make_fg(1, first_half, second_half);
      auto fg_d2 = make_fg(1, second_half, first_half);

      algorithms::ortools::SequenceFormLpSpecification sp_d1(
          *fg_d1, "CLP", false);
      sp_d1.SpecifyLinearProgram(1);
      sp_d1.Solve();
      TabularPolicy ppd1 = sp_d1.OptimalPolicy(1, true);

      algorithms::ortools::SequenceFormLpSpecification sp_d2(
          *fg_d2, "CLP", false);
      sp_d2.SpecifyLinearProgram(1);
      sp_d2.Solve();
      TabularPolicy ppd2 = sp_d2.OptimalPolicy(1, true);

      TabularPolicy stitch_d = full_ne;
      std::unordered_set<std::string> p1_allowed_a, p1_allowed_b;
      for (const auto& g : first_half) {
        auto it2 = decomp.grouped_subgames.find(g);
        if (it2 != decomp.grouped_subgames.end()) {
          auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
          p1_allowed_a.insert(info[1].begin(), info[1].end());
        }
      }
      for (const auto& g : second_half) {
        auto it2 = decomp.grouped_subgames.find(g);
        if (it2 != decomp.grouped_subgames.end()) {
          auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
          p1_allowed_b.insert(info[1].begin(), info[1].end());
        }
      }
      StitchPlayerFromFGPolicy(ppd1, 1, p1_allowed_a,
                               trunk_all_infosets, &stitch_d);
      StitchPlayerFromFGPolicy(ppd2, 1, p1_allowed_b,
                               trunk_all_infosets, &stitch_d);
      FillMissingFromReference(&stitch_d, full_ne,
                               all_is_orig[0], all_is_orig[1]);
      double exp_d = algorithms::Exploitability(*game, stitch_d);
      std::cout << "    P1-only rp=1 exp=" << exp_d << std::endl;
    }

    // Test E: P0 from rp=0 + P1 from rp=1 (replicates original experiment)
    std::cout << "  [Test E] P0 from rp=0 + P1 from rp=1 (full combo)"
              << std::endl;
    {
      auto fg_e0a = make_fg(0, first_half, second_half);
      auto fg_e0b = make_fg(0, second_half, first_half);
      auto fg_e1a = make_fg(1, first_half, second_half);
      auto fg_e1b = make_fg(1, second_half, first_half);

      auto solve_one = [](const Game& fg_game, int rp) {
        algorithms::ortools::SequenceFormLpSpecification sp(
            fg_game, "CLP", false);
        sp.SpecifyLinearProgram(rp);
        sp.Solve();
        return sp.OptimalPolicy(rp, true);
      };

      TabularPolicy p0a = solve_one(*fg_e0a, 0);
      TabularPolicy p0b = solve_one(*fg_e0b, 0);
      TabularPolicy p1a = solve_one(*fg_e1a, 1);
      TabularPolicy p1b = solve_one(*fg_e1b, 1);

      TabularPolicy stitch_e = full_ne;
      for (int pl = 0; pl < 2; ++pl) {
        for (const auto& half : {&first_half, &second_half}) {
          std::unordered_set<std::string> allowed;
          for (const auto& g : *half) {
            auto it2 = decomp.grouped_subgames.find(g);
            if (it2 != decomp.grouped_subgames.end()) {
              auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
              allowed.insert(info[pl].begin(), info[pl].end());
            }
          }
          const TabularPolicy& src = (pl == 0)
              ? (half == &first_half ? p0a : p0b)
              : (half == &first_half ? p1a : p1b);
          StitchPlayerFromFGPolicy(src, pl, allowed,
                                   trunk_all_infosets, &stitch_e);
        }
      }
      FillMissingFromReference(&stitch_e, full_ne,
                               all_is_orig[0], all_is_orig[1]);
      double exp_e = algorithms::Exploitability(*game, stitch_e);
      std::cout << "    P0(rp=0)+P1(rp=1) exp=" << exp_e << std::endl;

      // Test F/G (fixed scalar boundaries) disabled after boundary_values removal.
      if (false) {
      std::cout << "  [Test F] P1 from rp=1 with fixed-NE boundaries" << std::endl;
      {
        auto make_fg_fixed = [&](int rp,
                                 const std::vector<std::string>& target_groups,
                                 const std::vector<std::string>& other_groups) {
          std::unordered_map<std::string, std::vector<std::string>> bbg;
          std::unordered_map<std::string, std::vector<double>> bv;
          const std::string kT = "__gtarget__";
          const std::string kO = "__gother__";
          std::unordered_set<std::string> ts(target_groups.begin(),
                                             target_groups.end());
          for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
            const std::string bucket = ts.count(pub_obs) ? kT : kO;
            for (const auto& root : roots) {
              bbg[bucket].push_back(root->HistoryString());
              if (!ts.count(pub_obs)) {
                bv[root->HistoryString()] =
                    ComputeReturnsUnderPolicy(*root, full_ne);
              } else {
                bv[root->HistoryString()] = {0.0, 0.0};
              }
            }
          }
          return CreateFullGadgetGame(
              game, trunk_ptr, rp, kT, bbg, FullGadgetGame::Mode::kTrunk,
              {}, {}, /*enumerate_boundary_portfolios=*/false);
        };
        auto fg_g1 = make_fg_fixed(1, first_half, second_half);
        auto fg_g2 = make_fg_fixed(1, second_half, first_half);
        TabularPolicy ppg1 = solve_one(*fg_g1, 1);
        TabularPolicy ppg2 = solve_one(*fg_g2, 1);

        TabularPolicy stitch_g = full_ne;
        std::unordered_set<std::string> p1_ga, p1_gb;
        for (const auto& g : first_half) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            p1_ga.insert(info[1].begin(), info[1].end());
          }
        }
        for (const auto& g : second_half) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            p1_gb.insert(info[1].begin(), info[1].end());
          }
        }
        StitchPlayerFromFGPolicy(ppg1, 1, p1_ga, trunk_all_infosets, &stitch_g);
        StitchPlayerFromFGPolicy(ppg2, 1, p1_gb, trunk_all_infosets, &stitch_g);
        FillMissingFromReference(&stitch_g, full_ne, all_is_orig[0], all_is_orig[1]);
        double exp_g = algorithms::Exploitability(*game, stitch_g);
        std::cout << "    P1 fixed-NE rp=1 exp=" << exp_g << std::endl;
      }

      // Test G: P0 from rp=0 with fixed-NE boundaries (no MVS) as control
      std::cout << "  [Test G] P0 from rp=0 with fixed-NE boundaries" << std::endl;
      {
        auto make_fg_fixed = [&](int rp,
                                 const std::vector<std::string>& target_groups,
                                 const std::vector<std::string>& other_groups) {
          std::unordered_map<std::string, std::vector<std::string>> bbg;
          std::unordered_map<std::string, std::vector<double>> bv;
          const std::string kT = "__gtarget2__";
          const std::string kO = "__gother2__";
          std::unordered_set<std::string> ts(target_groups.begin(),
                                             target_groups.end());
          for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
            const std::string bucket = ts.count(pub_obs) ? kT : kO;
            for (const auto& root : roots) {
              bbg[bucket].push_back(root->HistoryString());
              if (!ts.count(pub_obs)) {
                bv[root->HistoryString()] =
                    ComputeReturnsUnderPolicy(*root, full_ne);
              } else {
                bv[root->HistoryString()] = {0.0, 0.0};
              }
            }
          }
          return CreateFullGadgetGame(
              game, trunk_ptr, rp, kT, bbg, FullGadgetGame::Mode::kTrunk,
              {}, {}, /*enumerate_boundary_portfolios=*/false);
        };
        auto fg_h1 = make_fg_fixed(0, first_half, second_half);
        auto fg_h2 = make_fg_fixed(0, second_half, first_half);
        TabularPolicy pph1 = solve_one(*fg_h1, 0);
        TabularPolicy pph2 = solve_one(*fg_h2, 0);

        TabularPolicy stitch_h = full_ne;
        std::unordered_set<std::string> p0_ha, p0_hb;
        for (const auto& g : first_half) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            p0_ha.insert(info[0].begin(), info[0].end());
          }
        }
        for (const auto& g : second_half) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            p0_hb.insert(info[0].begin(), info[0].end());
          }
        }
        StitchPlayerFromFGPolicy(pph1, 0, p0_ha, trunk_all_infosets, &stitch_h);
        StitchPlayerFromFGPolicy(pph2, 0, p0_hb, trunk_all_infosets, &stitch_h);
        FillMissingFromReference(&stitch_h, full_ne, all_is_orig[0], all_is_orig[1]);
        double exp_h = algorithms::Exploitability(*game, stitch_h);
        std::cout << "    P0 fixed-NE rp=0 exp=" << exp_h << std::endl;
      }
      }  // disabled fixed-scalar-boundary tests

      // Test H: Boundary MVS value verification
      std::cout << "  [Test H] Boundary MVS value check" << std::endl;
      {
        double max_value_gap = 0.0;
        int num_boundaries = 0;
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          for (const auto& root : roots) {
            auto ne_val = ComputeReturnsUnderPolicy(*root, full_ne);
            auto port_p0 = EnumerateSubtreePureStrategies(*root, 0);
            auto port_p1 = EnumerateSubtreePureStrategies(*root, 1);
            int k0 = port_p0.size();
            int k1 = port_p1.size();

            // Build payoff matrix (P1's payoff)
            std::vector<std::vector<double>> M(k0, std::vector<double>(k1));
            for (int i = 0; i < k0; ++i) {
              for (int j = 0; j < k1; ++j) {
                std::vector<const Policy*> pols = {port_p0[i].get(),
                                                    port_p1[j].get()};
                auto ret = algorithms::ExpectedReturns(
                    *root, pols, -1, false);
                M[i][j] = ret[1];
              }
            }

            // Solve matrix game: P1 maxmin via LP
            // max v s.t. for all i: sum_j M[i][j]*y[j] >= v, sum_j y[j]=1, y>=0
            auto lp = operations_research::MPSolver::CreateSolver("CLP");
            auto v_var = lp->MakeNumVar(-1e6, 1e6, "v");
            std::vector<operations_research::MPVariable*> y(k1);
            for (int j = 0; j < k1; ++j)
              y[j] = lp->MakeNumVar(0, 1, "");
            auto sum_ct = lp->MakeRowConstraint(1, 1);
            for (int j = 0; j < k1; ++j) sum_ct->SetCoefficient(y[j], 1);
            for (int i = 0; i < k0; ++i) {
              auto ct = lp->MakeRowConstraint(0, 1e6);
              ct->SetCoefficient(v_var, -1);
              for (int j = 0; j < k1; ++j)
                ct->SetCoefficient(y[j], M[i][j]);
            }
            lp->MutableObjective()->SetCoefficient(v_var, 1);
            lp->MutableObjective()->SetMaximization();
            lp->Solve();
            double mg_val = v_var->solution_value();

            double gap = std::abs(mg_val - ne_val[1]);
            max_value_gap = std::max(max_value_gap, gap);
            num_boundaries++;

            if (gap > 1e-6) {
              std::cout << "    MISMATCH at " << root->HistoryString()
                        << ": NE=" << ne_val[1] << " MG=" << mg_val
                        << " gap=" << gap
                        << " K0=" << k0 << " K1=" << k1 << std::endl;
            }
          }
        }
        std::cout << "    Checked " << num_boundaries
                  << " boundaries, max_value_gap=" << max_value_gap << std::endl;
      }

      // Test I: LP value comparison
      std::cout << "  [Test I] LP value comparison" << std::endl;
      {
        for (int rp = 0; rp < 2; ++rp) {
          for (const auto* halves : {&first_half, &second_half}) {
            const auto* other = (halves == &first_half) ? &second_half : &first_half;
            auto fg = make_fg(rp, *halves, *other);
            algorithms::ortools::SequenceFormLpSpecification sp(*fg, "CLP", false);
            sp.SpecifyLinearProgram(rp);
            double v = sp.Solve();
            std::cout << "    rp=" << rp
                      << " target=" << (halves == &first_half ? "first" : "second")
                      << std::setprecision(15) << " LP_val=" << v << std::endl;
          }
        }
      }

      // Test J: Strategy comparison – exactly which info states differ
      // between the LP-derived strategy and the full NE, for each rp.
      std::cout << "  [Test J] Per-info-state strategy comparison" << std::endl;
      {
        for (int rp = 0; rp < 2; ++rp) {
          auto fg_j = make_fg(rp, first_half, second_half);
          std::cout << "    rp=" << rp << std::endl;

          algorithms::ortools::SequenceFormLpSpecification sp_j(
              *fg_j, "CLP", false);
          sp_j.SpecifyLinearProgram(rp);
          double lp_val = sp_j.Solve();
          TabularPolicy pp_rp = sp_j.OptimalPolicy(rp, true);
          std::cout << "      LP val=" << lp_val << std::endl;

          // Stitch the resolving player's target-subgame strategy
          TabularPolicy stitch_j = full_ne;
          std::unordered_set<std::string> allowed_j;
          for (const auto& g : first_half) {
            auto it2 = decomp.grouped_subgames.find(g);
            if (it2 != decomp.grouped_subgames.end()) {
              auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
              allowed_j.insert(info[rp].begin(), info[rp].end());
            }
          }
          StitchPlayerFromFGPolicy(pp_rp, rp, allowed_j,
                                   trunk_all_infosets, &stitch_j);
          FillMissingFromReference(&stitch_j, full_ne,
                                   all_is_orig[0], all_is_orig[1]);
          double stitch_exp = algorithms::Exploitability(*game, stitch_j);
          std::cout << "      Stitched exp=" << stitch_exp << std::endl;

          // Compare: for each info state of the resolving player in the
          // target subgame, is the LP strategy the same as the NE?
          int n_diff = 0, n_same = 0, n_miss = 0;
          for (const auto& is_str : allowed_j) {
            auto ne_ap = full_ne.GetStatePolicy(is_str);
            auto lp_ap = stitch_j.GetStatePolicy(is_str);
            if (ne_ap.empty()) { n_miss++; continue; }
            if (lp_ap.empty()) { n_miss++; continue; }
            bool same = true;
            for (size_t k = 0; k < ne_ap.size(); ++k) {
              if (std::abs(ne_ap[k].second - lp_ap[k].second) > 1e-9) {
                same = false;
                break;
              }
            }
            if (same) { n_same++; }
            else {
              n_diff++;
              if (n_diff <= 8) {
                std::cout << "        DIFF is=" << is_str << std::endl;
                std::cout << "          NE:";
                for (auto& [a,p] : ne_ap) std::cout << " " << a << ":" << p;
                std::cout << std::endl;
                std::cout << "          LP:";
                for (auto& [a,p] : lp_ap) std::cout << " " << a << ":" << p;
                std::cout << std::endl;
              }
            }
          }
          std::cout << "      same=" << n_same << " diff=" << n_diff
                    << " miss=" << n_miss << std::endl;

          // Also solve SEPARATELY for the opponent to get their minimax
          // strategy in the same FG game structure.
          algorithms::ortools::SequenceFormLpSpecification sp_opp(
              *fg_j, "CLP", false);
          sp_opp.SpecifyLinearProgram(1-rp);
          sp_opp.Solve();
          TabularPolicy pp_opp = sp_opp.OptimalPolicy(1-rp, true);
          std::unordered_set<std::string> opp_allowed;
          for (const auto& g : first_half) {
            auto it2 = decomp.grouped_subgames.find(g);
            if (it2 != decomp.grouped_subgames.end()) {
              auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
              opp_allowed.insert(info[1-rp].begin(), info[1-rp].end());
            }
          }
          TabularPolicy opp_stitch = full_ne;
          StitchPlayerFromFGPolicy(pp_opp, 1-rp, opp_allowed,
                                   trunk_all_infosets, &opp_stitch);
          FillMissingFromReference(&opp_stitch, full_ne,
                                   all_is_orig[0], all_is_orig[1]);
          double opp_exp = algorithms::Exploitability(*game, opp_stitch);
          std::cout << "      Opponent-only stitch (pl=" << 1-rp
                    << ") exp=" << opp_exp << std::endl;
          int opp_diff = 0, opp_same = 0;
          for (const auto& is_str : opp_allowed) {
            auto ne_ap = full_ne.GetStatePolicy(is_str);
            auto br_ap = opp_stitch.GetStatePolicy(is_str);
            if (ne_ap.empty() || br_ap.empty()) continue;
            bool same = true;
            for (size_t k = 0; k < ne_ap.size(); ++k) {
              if (std::abs(ne_ap[k].second - br_ap[k].second) > 1e-9) {
                same = false; break;
              }
            }
            if (same) opp_same++; else opp_diff++;
          }
          std::cout << "      Opponent diff vs NE: same=" << opp_same
                    << " diff=" << opp_diff << std::endl;
        }
      }

      // Test K: investigate rp=1 mismatch regions and boundary abstraction loss.
      std::cout << "  [Test K] rp=1 mismatch vs boundary abstraction detail"
                << std::endl;
      {
        auto fg_k = make_fg(1, first_half, second_half);
        algorithms::ortools::SequenceFormLpSpecification sp_k(
            *fg_k, "CLP", false);
        sp_k.SpecifyLinearProgram(1);
        sp_k.Solve();
        TabularPolicy pp_k = sp_k.OptimalPolicy(1, true);

        // Build stitched policy for P1 on first-half targets only.
        TabularPolicy stitch_k = full_ne;
        std::unordered_set<std::string> allowed_k;
        for (const auto& g : first_half) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            allowed_k.insert(info[1].begin(), info[1].end());
          }
        }
        StitchPlayerFromFGPolicy(pp_k, 1, allowed_k, trunk_all_infosets, &stitch_k);
        FillMissingFromReference(&stitch_k, full_ne, all_is_orig[0], all_is_orig[1]);

        // Count where P1 differs from NE by public observation tag.
        std::map<std::string, int> diff_by_public;
        int total_diff = 0;
        for (const auto& is_str : allowed_k) {
          auto ne_ap = full_ne.GetStatePolicy(is_str);
          auto lp_ap = stitch_k.GetStatePolicy(is_str);
          if (ne_ap.empty() || lp_ap.empty()) continue;
          bool same = true;
          for (size_t k = 0; k < ne_ap.size(); ++k) {
            if (std::abs(ne_ap[k].second - lp_ap[k].second) > 1e-9) {
              same = false;
              break;
            }
          }
          if (!same) {
            total_diff++;
            diff_by_public[ExtractPublicTag(is_str)]++;
          }
        }
        std::cout << "    P1 differing infosets in first-half target: "
                  << total_diff << std::endl;
        for (const auto& [pub, cnt] : diff_by_public) {
          std::cout << "      public=" << pub << " diff_count=" << cnt
                    << std::endl;
        }

        // For the collapsed half, compare full subtree detail vs matrix game.
        std::unordered_set<std::string> second_set(second_half.begin(),
                                                   second_half.end());
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          if (second_set.count(pub_obs) == 0) continue;
          for (const auto& root : roots) {
            std::vector<std::unique_ptr<State>> single_root;
            single_root.push_back(root->Clone());
            auto info_sets = CollectSubgameInfoStatesPerPlayer(single_root);
            int n_t = CountTerminalStates(*root);

            auto port_p0 = EnumerateSubtreePureStrategies(*root, 0);
            auto port_p1 = EnumerateSubtreePureStrategies(*root, 1);
            int k0 = port_p0.size();
            int k1 = port_p1.size();
            double min_pay = 1e9, max_pay = -1e9;
            std::vector<std::vector<double>> M(k0, std::vector<double>(k1));
            for (int i = 0; i < k0; ++i) {
              for (int j = 0; j < k1; ++j) {
                std::vector<const Policy*> pols = {port_p0[i].get(),
                                                   port_p1[j].get()};
                auto ret = algorithms::ExpectedReturns(*root, pols, -1, false);
                M[i][j] = ret[1];
                min_pay = std::min(min_pay, ret[1]);
                max_pay = std::max(max_pay, ret[1]);
              }
            }
            auto lp = operations_research::MPSolver::CreateSolver("CLP");
            auto v_var = lp->MakeNumVar(-1e6, 1e6, "v");
            std::vector<operations_research::MPVariable*> y(k1);
            for (int j = 0; j < k1; ++j) y[j] = lp->MakeNumVar(0, 1, "");
            auto sum_ct = lp->MakeRowConstraint(1, 1);
            for (int j = 0; j < k1; ++j) sum_ct->SetCoefficient(y[j], 1);
            for (int i = 0; i < k0; ++i) {
              auto ct = lp->MakeRowConstraint(0, 1e6);
              ct->SetCoefficient(v_var, -1);
              for (int j = 0; j < k1; ++j) ct->SetCoefficient(y[j], M[i][j]);
            }
            lp->MutableObjective()->SetCoefficient(v_var, 1);
            lp->MutableObjective()->SetMaximization();
            lp->Solve();
            double mg_val = v_var->solution_value();
            auto ne_val = ComputeReturnsUnderPolicy(*root, full_ne);

            std::string pub = ExtractPublicTag(root->InformationStateString(1));
            bool highlighted = (diff_by_public.count(pub) > 0);
            if (highlighted) {
              std::cout << "    COLLAPSED boundary root hist="
                        << root->HistoryString() << std::endl;
              std::cout << "      public=" << pub
                        << " p0_infosets=" << info_sets[0].size()
                        << " p1_infosets=" << info_sets[1].size()
                        << " terminals=" << n_t << std::endl;
              std::cout << "      matrix k0=" << k0 << " k1=" << k1
                        << " payoff_range=[" << min_pay << "," << max_pay
                        << "]" << std::endl;
              std::cout << "      value compare P1: NE=" << ne_val[1]
                        << " MG_maxmin=" << mg_val
                        << " gap=" << std::abs(ne_val[1] - mg_val)
                        << std::endl;
            }
          }
        }
      }
    }
  }

  // Test L: Verify matrix game correctness (no trunk-solve info) and baseline.
  // 1) Check BI(root) == MG_maxmin(root) for every boundary root.
  // 2) All-targets FG game (no matrix boundary) → stitch → exp should be ~0.
  // 3) Single-matrix-boundary FG game → stitch → shows per-group effect.
  std::cout << "\n=== [Test L] Matrix-game verification + all-targets baseline ==="
            << std::endl;
  {
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    auto all_is_orig = CollectAllInfoStates(*game);

    // Exact subgame solve at a single root (imperfect-information correct):
    // build a one-root unsafe subgame and solve via sequence-form LP.
    auto exact_root_value_p1 = [&](const State& root) -> double {
      std::vector<std::unique_ptr<State>> one_root;
      one_root.push_back(root.Clone());
      std::vector<double> reach = {1.0};
      auto one_root_game =
          CreateUnsafeSubgame(game, std::move(one_root), std::move(reach));
      auto [eq_pol, eq_val] =
          algorithms::ortools::MakeEquilibriumPolicy(*one_root_game, true);
      auto init = one_root_game->NewInitialState();
      auto ret = algorithms::ExpectedReturns(*init, eq_pol, -1, true);
      return ret[1];
    };

    // (L-1) exact one-root LP solve vs matrix-game maxmin.
    std::cout << "  [L-1] Per-root exact subgame LP vs matrix-game maxmin"
              << std::endl;
    int total_roots = 0, mismatches = 0;
    double max_gap = 0.0;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        double exact_val = exact_root_value_p1(*root);

        auto port_p0 = EnumerateSubtreePureStrategies(*root, 0);
        auto port_p1 = EnumerateSubtreePureStrategies(*root, 1);
        int k0 = port_p0.size();
        int k1 = port_p1.size();

        std::vector<std::vector<double>> M(k0, std::vector<double>(k1));
        for (int i = 0; i < k0; ++i) {
          for (int j = 0; j < k1; ++j) {
            std::vector<const Policy*> pols = {port_p0[i].get(),
                                               port_p1[j].get()};
            auto ret = algorithms::ExpectedReturns(*root, pols, -1, false);
            M[i][j] = ret[1];
          }
        }

        auto lp = operations_research::MPSolver::CreateSolver("CLP");
        auto v_var = lp->MakeNumVar(-1e6, 1e6, "v");
        std::vector<operations_research::MPVariable*> y(k1);
        for (int j = 0; j < k1; ++j) y[j] = lp->MakeNumVar(0, 1, "");
        auto sum_ct = lp->MakeRowConstraint(1, 1);
        for (int j = 0; j < k1; ++j) sum_ct->SetCoefficient(y[j], 1);
        for (int i = 0; i < k0; ++i) {
          auto ct = lp->MakeRowConstraint(0, 1e6);
          ct->SetCoefficient(v_var, -1);
          for (int j = 0; j < k1; ++j) ct->SetCoefficient(y[j], M[i][j]);
        }
        lp->MutableObjective()->SetCoefficient(v_var, 1);
        lp->MutableObjective()->SetMaximization();
        lp->Solve();
        double mg_val = v_var->solution_value();

        double gap = std::abs(exact_val - mg_val);
        max_gap = std::max(max_gap, gap);
        total_roots++;
        if (gap > 1e-8) {
          mismatches++;
          std::cout << "    MISMATCH root=" << root->HistoryString()
                    << " exact=" << exact_val << " MG=" << mg_val
                    << " gap=" << gap << " K0=" << k0 << " K1=" << k1
                    << std::endl;
        }
      }
    }
    std::cout << "    total_roots=" << total_roots
              << " mismatches=" << mismatches
              << " max_gap=" << max_gap << std::endl;

    // (L-2) All-targets baseline: FG game with no matrix boundary at all.
    std::cout << "  [L-2] All-targets baseline (no matrix boundary)" << std::endl;
    for (int rp = 0; rp < 2; ++rp) {
      std::unordered_map<std::string, std::vector<std::string>> bbg;
      std::unordered_map<std::string, std::vector<double>> bv;
      const std::string kAll = "__all_tgt__";
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        for (const auto& root : roots) {
          bbg[kAll].push_back(root->HistoryString());
          bv[root->HistoryString()] = {0.0, 0.0};
        }
      }
      auto fg_all = CreateFullGadgetGame(
          game, trunk_ptr, rp, kAll, bbg,
          FullGadgetGame::Mode::kTrunk,
          {}, {}, /*enumerate_boundary_portfolios=*/false);

      algorithms::ortools::SequenceFormLpSpecification sp_all(
          *fg_all, "CLP", false);
      sp_all.SpecifyLinearProgram(rp);
      double v_all = sp_all.Solve();
      TabularPolicy pp_all = sp_all.OptimalPolicy(rp, true);

      TabularPolicy stitch_all = full_ne;
      std::unordered_set<std::string> allowed_all;
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        auto info = CollectSubgameInfoStatesPerPlayer(roots);
        allowed_all.insert(info[rp].begin(), info[rp].end());
      }
      StitchPlayerFromFGPolicy(pp_all, rp, allowed_all,
                               trunk_all_infosets, &stitch_all);
      FillMissingFromReference(&stitch_all, full_ne,
                               all_is_orig[0], all_is_orig[1]);
      double exp_all = algorithms::Exploitability(*game, stitch_all);
      std::cout << "    rp=" << rp << " LP_val=" << v_all
                << " exp=" << exp_all << std::endl;
    }

    // (L-3) Single-group matrix boundary, rest as targets.
    if (groups.size() >= 2) {
      std::cout << "  [L-3] Single-group matrix boundary" << std::endl;
      std::string single_bnd = groups[0];
      std::vector<std::string> rest_tgt(groups.begin() + 1, groups.end());

      for (int rp = 0; rp < 2; ++rp) {
        std::unordered_map<std::string, std::vector<std::string>> bbg;
        std::unordered_map<std::string, std::vector<double>> bv;
        const std::string kT = "__sgl_tgt__";
        const std::string kO = "__sgl_bnd__";
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          const std::string bucket =
              (pub_obs == single_bnd) ? kO : kT;
          for (const auto& root : roots) {
            bbg[bucket].push_back(root->HistoryString());
            bv[root->HistoryString()] = {0.0, 0.0};
          }
        }
        auto fg_sgl = CreateFullGadgetGame(
            game, trunk_ptr, rp, kT, bbg,
            FullGadgetGame::Mode::kTrunk,
            {}, {}, /*enumerate_boundary_portfolios=*/true);

        algorithms::ortools::SequenceFormLpSpecification sp_sgl(
            *fg_sgl, "CLP", false);
        sp_sgl.SpecifyLinearProgram(rp);
        double v_sgl = sp_sgl.Solve();
        TabularPolicy pp_sgl = sp_sgl.OptimalPolicy(rp, true);

        TabularPolicy stitch_sgl = full_ne;
        std::unordered_set<std::string> allowed_sgl;
        for (const auto& g : rest_tgt) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 != decomp.grouped_subgames.end()) {
            auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
            allowed_sgl.insert(info[rp].begin(), info[rp].end());
          }
        }
        StitchPlayerFromFGPolicy(pp_sgl, rp, allowed_sgl,
                                 trunk_all_infosets, &stitch_sgl);
        FillMissingFromReference(&stitch_sgl, full_ne,
                                 all_is_orig[0], all_is_orig[1]);
        double exp_sgl = algorithms::Exploitability(*game, stitch_sgl);
        std::cout << "    rp=" << rp << " boundary_group=" << single_bnd
                  << " LP_val=" << v_sgl << " exp=" << exp_sgl << std::endl;
      }
    }
  }

  // Test M: Solve FG with half targets, then compare target-root continuation
  // values per infoset against full-game NE.
  std::cout << "\n=== [Test M] Half-target FG root values per infoset vs full NE ==="
            << std::endl;
  {
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    auto all_is_orig = CollectAllInfoStates(*game);

    auto make_fg = [&](int rp,
                       const std::vector<std::string>& target_groups,
                       const std::vector<std::string>& other_groups) {
      std::unordered_map<std::string, std::vector<std::string>> bbg;
      std::unordered_map<std::string, std::vector<double>> bv;
      const std::string kT = "__mtarget__";
      const std::string kO = "__mother__";
      std::unordered_set<std::string> ts(target_groups.begin(),
                                         target_groups.end());
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        const std::string bucket = ts.count(pub_obs) ? kT : kO;
        for (const auto& root : roots) {
          bbg[bucket].push_back(root->HistoryString());
          bv[root->HistoryString()] = {0.0, 0.0};
        }
      }
      return CreateFullGadgetGame(
          game, trunk_ptr, rp, kT, bbg, FullGadgetGame::Mode::kTrunk,
          {}, {}, /*enumerate_boundary_portfolios=*/true);
    };

    auto compare_half = [&](int rp,
                            const std::vector<std::string>& target_groups,
                            const std::string& label) {
      std::cout << "  [M] rp=" << rp << " target=" << label << std::endl;
      auto fg = make_fg(rp, target_groups,
                        (label == "first") ? second_half : first_half);
      algorithms::ortools::SequenceFormLpSpecification sp_r(*fg, "CLP", false);
      sp_r.SpecifyLinearProgram(rp);
      double fg_val_r = sp_r.Solve();
      TabularPolicy fg_pol_r = sp_r.OptimalPolicy(rp, true);
      algorithms::ortools::SequenceFormLpSpecification sp_o(*fg, "CLP", false);
      sp_o.SpecifyLinearProgram(1 - rp);
      double fg_val_o = sp_o.Solve();
      TabularPolicy fg_pol_o = sp_o.OptimalPolicy(1 - rp, true);
      std::cout << "    FG per-player LP values rp/opp=" << fg_val_r
                << " / " << fg_val_o << std::endl;

      // Build an original-game policy where only target subgame rows are taken
      // from FG, and everything else is from full-game NE.
      TabularPolicy stitched = full_ne;
      for (int pl = 0; pl < 2; ++pl) {
        std::unordered_set<std::string> allowed;
        for (const auto& g : target_groups) {
          auto it2 = decomp.grouped_subgames.find(g);
          if (it2 == decomp.grouped_subgames.end()) continue;
          auto info = CollectSubgameInfoStatesPerPlayer(it2->second);
          allowed.insert(info[pl].begin(), info[pl].end());
        }
        const TabularPolicy& src = (pl == rp) ? fg_pol_r : fg_pol_o;
        StitchPlayerFromFGPolicy(src, pl, allowed, trunk_all_infosets, &stitched);
      }
      FillMissingFromReference(&stitched, full_ne, all_is_orig[0], all_is_orig[1]);

      // Compare continuation values at every root in the target half.
      double max_gap_p0 = 0.0, max_gap_p1 = 0.0;
      double sum_gap_p0 = 0.0, sum_gap_p1 = 0.0;
      int n_roots = 0;
      int num_large = 0;
      std::unordered_map<std::string, std::pair<double, int>> p0_is_gap;
      std::unordered_map<std::string, std::pair<double, int>> p1_is_gap;

      std::unordered_set<std::string> tset(target_groups.begin(),
                                           target_groups.end());
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        if (tset.count(pub_obs) == 0) continue;
        for (const auto& root : roots) {
          auto v_full = ComputeReturnsUnderPolicy(*root, full_ne);
          auto v_st = ComputeReturnsUnderPolicy(*root, stitched);
          double g0 = std::abs(v_full[0] - v_st[0]);
          double g1 = std::abs(v_full[1] - v_st[1]);
          max_gap_p0 = std::max(max_gap_p0, g0);
          max_gap_p1 = std::max(max_gap_p1, g1);
          sum_gap_p0 += g0;
          sum_gap_p1 += g1;
          ++n_roots;

          std::string is0 = root->InformationStateString(0);
          std::string is1 = root->InformationStateString(1);
          p0_is_gap[is0].first += g0;
          p0_is_gap[is0].second += 1;
          p1_is_gap[is1].first += g1;
          p1_is_gap[is1].second += 1;

          if (g0 > 1e-6 || g1 > 1e-6) {
            ++num_large;
            if (num_large <= 8) {
              std::cout << "      root=" << root->HistoryString() << std::endl;
              std::cout << "        P0 full/stitch/gap: " << v_full[0] << " / "
                        << v_st[0] << " / " << g0 << std::endl;
              std::cout << "        P1 full/stitch/gap: " << v_full[1] << " / "
                        << v_st[1] << " / " << g1 << std::endl;
            }
          }
        }
      }

      std::cout << "    roots=" << n_roots
                << " max_gap(P0/P1)=" << max_gap_p0 << "/" << max_gap_p1
                << " mean_gap(P0/P1)="
                << (n_roots > 0 ? sum_gap_p0 / n_roots : 0.0) << "/"
                << (n_roots > 0 ? sum_gap_p1 / n_roots : 0.0)
                << " large_gap_roots=" << num_large << std::endl;

      auto print_top_infoset_gaps =
          [](const std::unordered_map<std::string, std::pair<double, int>>& m,
             int player) {
            std::vector<std::pair<double, std::string>> rows;
            rows.reserve(m.size());
            for (const auto& [is, s] : m) {
              if (s.second <= 0) continue;
              rows.push_back({s.first / s.second, is});
            }
            std::sort(rows.begin(), rows.end(),
                      [](const auto& a, const auto& b) {
                        return a.first > b.first;
                      });
            std::cout << "    top infoset mean value gaps P" << player << ":"
                      << std::endl;
            for (int i = 0; i < std::min(6, static_cast<int>(rows.size())); ++i) {
              std::cout << "      [" << (i + 1) << "] gap=" << rows[i].first
                        << " is=" << rows[i].second << std::endl;
            }
          };

      print_top_infoset_gaps(p0_is_gap, 0);
      print_top_infoset_gaps(p1_is_gap, 1);
    };

    compare_half(0, first_half, "first");
    compare_half(0, second_half, "second");
    compare_half(1, first_half, "first");
    compare_half(1, second_half, "second");
  }

  // Test N: print one concrete root subtree and its matrix representation.
  std::cout << "\n=== [Test N] One concrete root: tree and matrix ===" << std::endl;
  {
    const State* picked = nullptr;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        if (root->HistoryString() == "5, 1, 2, 1, 0") {
          picked = root.get();
          break;
        }
      }
      if (picked != nullptr) break;
    }
    if (picked == nullptr) {
      auto itg = decomp.grouped_subgames.begin();
      if (itg != decomp.grouped_subgames.end() && !itg->second.empty()) {
        picked = itg->second[0].get();
      }
    }
    SPIEL_CHECK_TRUE(picked != nullptr);
    PrintRootTreeAndMatrix(*picked);
  }

  // Test O: all roots - compare tree terminal payoff set vs matrix payoff set.
  std::cout << "\n=== [Test O] All roots: tree terminal values vs matrix values ==="
            << std::endl;
  {
    int roots_total = 0;
    int roots_match_full = 0;
    int roots_match_reduced = 0;
    int roots_mismatch_full = 0;
    int roots_mismatch_reduced = 0;
    int printed = 0;

    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) {
        ++roots_total;
        std::vector<double> terminal_vals;
        CollectTerminalPayoffsP1(*root, &terminal_vals);
        auto term_set = UniqueSortedTol(std::move(terminal_vals));

        auto p0_full = EnumerateSubtreePureStrategies(*root, 0);
        auto p1_full = EnumerateSubtreePureStrategies(*root, 1);
        std::vector<double> matrix_vals_full;
        matrix_vals_full.reserve(p0_full.size() * p1_full.size());
        for (int i = 0; i < static_cast<int>(p0_full.size()); ++i) {
          for (int j = 0; j < static_cast<int>(p1_full.size()); ++j) {
            std::vector<const Policy*> pols = {p0_full[i].get(), p1_full[j].get()};
            auto ret = algorithms::ExpectedReturns(
                *root, pols, /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
            matrix_vals_full.push_back(ret[1]);
          }
        }
        auto mat_set_full = UniqueSortedTol(std::move(matrix_vals_full));
        bool ok_full = SetEqualTol(term_set, mat_set_full);
        if (ok_full) ++roots_match_full;
        else ++roots_mismatch_full;

        auto p0_red = EnumerateSubtreePureStrategiesReduced(*root, 0);
        auto p1_red = EnumerateSubtreePureStrategiesReduced(*root, 1);
        std::vector<double> matrix_vals_red;
        matrix_vals_red.reserve(p0_red.size() * p1_red.size());
        for (int i = 0; i < static_cast<int>(p0_red.size()); ++i) {
          for (int j = 0; j < static_cast<int>(p1_red.size()); ++j) {
            std::vector<const Policy*> pols = {p0_red[i].get(), p1_red[j].get()};
            auto ret = algorithms::ExpectedReturns(
                *root, pols, /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
            matrix_vals_red.push_back(ret[1]);
          }
        }
        auto mat_set_red = UniqueSortedTol(std::move(matrix_vals_red));
        bool ok_red = SetEqualTol(term_set, mat_set_red);
        if (ok_red) ++roots_match_reduced;
        else ++roots_mismatch_reduced;

        if ((!ok_full || !ok_red) && printed < 8) {
          ++printed;
          std::cout << "  MISMATCH root=" << root->HistoryString()
                    << " pub=" << pub_obs << std::endl;
          std::cout << "    tree terminals (P1): " << VecToString(term_set) << std::endl;
          std::cout << "    matrix full (P1):    " << VecToString(mat_set_full)
                    << " K0xK1=" << p0_full.size() << "x" << p1_full.size()
                    << std::endl;
          std::cout << "    matrix reduced (P1): " << VecToString(mat_set_red)
                    << " K0xK1=" << p0_red.size() << "x" << p1_red.size()
                    << std::endl;
        }
      }
    }

    std::cout << "  roots_total=" << roots_total << std::endl;
    std::cout << "  full matrix set match/mismatch="
              << roots_match_full << "/" << roots_mismatch_full << std::endl;
    std::cout << "  reduced matrix set match/mismatch="
              << roots_match_reduced << "/" << roots_mismatch_reduced << std::endl;
  }

  // Test P: collapse exactly ONE group to MVS, keep all others as full target.
  // Run for every group and both players; stitch back into full NE and measure
  // exploitability. This isolates whether a single MVS group alone causes issues.
  std::cout << "\n=== [Test P] One-MVS-group sweep (all groups, both players) ==="
            << std::endl;
  {
    auto all_is_orig = CollectAllInfoStates(*game);
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    const double report_threshold = 1e-4;

    std::vector<std::pair<std::string, double>> per_group_exp;
    per_group_exp.reserve(groups.size());
    std::vector<std::pair<std::string, double>> per_group_exp_p0_only;
    std::vector<std::pair<std::string, double>> per_group_exp_p1_only;
    per_group_exp_p0_only.reserve(groups.size());
    per_group_exp_p1_only.reserve(groups.size());

    for (const auto& mvs_group : groups) {
      // Build two buckets: target=all groups except mvs_group, other=mvs_group.
      const std::string kTarget = "__one_mvs_target__";
      const std::string kOther = "__one_mvs_other__";
      std::unordered_map<std::string, std::vector<std::string>> bbg;
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        const std::string bucket = (pub_obs == mvs_group) ? kOther : kTarget;
        for (const auto& root : roots) bbg[bucket].push_back(root->HistoryString());
      }

      auto make_fg_for_rp = [&](int rp) {
        return CreateFullGadgetGame(
            game, trunk_ptr, rp, kTarget, bbg, FullGadgetGame::Mode::kTrunk,
            {}, {}, /*enumerate_boundary_portfolios=*/true);
      };

      auto solve_one = [](const Game& fg_game, int rp) {
        algorithms::ortools::SequenceFormLpSpecification sp(fg_game, "CLP", false);
        sp.SpecifyLinearProgram(rp);
        double v = sp.Solve();
        return std::make_pair(v, sp.OptimalPolicy(rp, true));
      };

      auto fg0 = make_fg_for_rp(0);
      auto fg1 = make_fg_for_rp(1);
      auto [v0, pol0] = solve_one(*fg0, 0);
      auto [v1, pol1] = solve_one(*fg1, 1);

      // Allowed infosets are all groups except the one collapsed to MVS.
      std::unordered_set<std::string> allowed0, allowed1;
      for (const auto& g : groups) {
        if (g == mvs_group) continue;
        auto itg = decomp.grouped_subgames.find(g);
        if (itg == decomp.grouped_subgames.end()) continue;
        auto info = CollectSubgameInfoStatesPerPlayer(itg->second);
        allowed0.insert(info[0].begin(), info[0].end());
        allowed1.insert(info[1].begin(), info[1].end());
      }

      TabularPolicy stitched = full_ne;
      StitchPlayerFromFGPolicy(pol0, 0, allowed0, trunk_all_infosets, &stitched);
      StitchPlayerFromFGPolicy(pol1, 1, allowed1, trunk_all_infosets, &stitched);
      FillMissingFromReference(&stitched, full_ne, all_is_orig[0], all_is_orig[1]);
      double exp = algorithms::Exploitability(*game, stitched);
      per_group_exp.push_back({mvs_group, exp});

      TabularPolicy stitched_p0 = full_ne;
      StitchPlayerFromFGPolicy(pol0, 0, allowed0, trunk_all_infosets, &stitched_p0);
      FillMissingFromReference(&stitched_p0, full_ne, all_is_orig[0], all_is_orig[1]);
      double exp_p0 = algorithms::Exploitability(*game, stitched_p0);
      per_group_exp_p0_only.push_back({mvs_group, exp_p0});

      TabularPolicy stitched_p1 = full_ne;
      StitchPlayerFromFGPolicy(pol1, 1, allowed1, trunk_all_infosets, &stitched_p1);
      FillMissingFromReference(&stitched_p1, full_ne, all_is_orig[0], all_is_orig[1]);
      double exp_p1 = algorithms::Exploitability(*game, stitched_p1);
      per_group_exp_p1_only.push_back({mvs_group, exp_p1});

      std::cout << "  mvs_group=" << mvs_group
                << " lp_vals(P0/P1)=" << v0 << "/" << v1
                << " stitched_exp=" << exp
                << " p0_only_exp=" << exp_p0
                << " p1_only_exp=" << exp_p1
                << std::endl;
    }

    std::sort(per_group_exp.begin(), per_group_exp.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    double sum = 0.0;
    int num_gt_1e6 = 0, num_gt_1e4 = 0;
    for (const auto& [g, e] : per_group_exp) {
      sum += e;
      if (e > 1e-6) ++num_gt_1e6;
      if (e > 1e-4) ++num_gt_1e4;
    }
    double mean = per_group_exp.empty() ? 0.0 : sum / per_group_exp.size();
    double max_e = per_group_exp.empty() ? 0.0 : per_group_exp.front().second;
    double min_e = per_group_exp.empty() ? 0.0 : per_group_exp.back().second;
    std::cout << "  Summary one-MVS-group: n=" << per_group_exp.size()
              << " mean=" << mean
              << " min=" << min_e
              << " max=" << max_e
              << " >1e-6=" << num_gt_1e6
              << " >1e-4=" << num_gt_1e4
              << std::endl;
    std::cout << "  Top-5 worst groups:" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(per_group_exp.size())); ++i) {
      std::cout << "    [" << (i + 1) << "] exp=" << per_group_exp[i].second
                << " group=" << per_group_exp[i].first << std::endl;
    }

    auto summarize_one_player =
        [&](const std::vector<std::pair<std::string, double>>& rows,
            const std::string& label) {
          std::vector<std::pair<std::string, double>> sorted = rows;
          std::sort(sorted.begin(), sorted.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
          double s = 0.0;
          int above = 0;
          for (const auto& [g, e] : sorted) {
            s += e;
            if (e > report_threshold) ++above;
          }
          double m = sorted.empty() ? 0.0 : s / sorted.size();
          double mx = sorted.empty() ? 0.0 : sorted.front().second;
          double mn = sorted.empty() ? 0.0 : sorted.back().second;
          std::cout << "  Summary " << label
                    << ": n=" << sorted.size()
                    << " mean=" << m
                    << " min=" << mn
                    << " max=" << mx
                    << " >" << report_threshold << "=" << above
                    << std::endl;
          std::cout << "  " << label << " groups above threshold:" << std::endl;
          int printed_local = 0;
          for (const auto& [g, e] : sorted) {
            if (e <= report_threshold) continue;
            ++printed_local;
            std::cout << "    [" << printed_local << "] exp=" << e
                      << " group=" << g << std::endl;
          }
          if (printed_local == 0) std::cout << "    (none)" << std::endl;
        };

    summarize_one_player(per_group_exp_p0_only, "P0-only one-MVS-group");
    summarize_one_player(per_group_exp_p1_only, "P1-only one-MVS-group");

    // Detailed dump: print trees and matrices for P1-only groups above threshold.
    std::vector<std::string> flagged_p1_groups;
    {
      std::vector<std::pair<std::string, double>> sorted = per_group_exp_p1_only;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
      for (const auto& [g, e] : sorted) {
        if (e > report_threshold) flagged_p1_groups.push_back(g);
      }
    }
    std::cout << "\n=== [Test Q] Detailed trees+matrices for flagged P1-only groups ==="
              << std::endl;
    for (const auto& g : flagged_p1_groups) {
      auto itg = decomp.grouped_subgames.find(g);
      if (itg == decomp.grouped_subgames.end()) continue;
      std::cout << "\n  GROUP: " << g << " roots=" << itg->second.size()
                << std::endl;
      int ridx = 0;
      for (const auto& root : itg->second) {
        ++ridx;
        std::cout << "\n  -- root " << ridx << "/" << itg->second.size()
                  << " --" << std::endl;
        PrintRootTreeAndMatrix(*root);
      }
    }
  }

  // Test R: Stitching-only diagnostic (no MVS).
  // Solve full-game LP and select different NE points via tie-break objectives.
  // Then stitch by whole public-state groups and measure exploitability.
  std::cout << "\n=== [Test R] Stitching-only with different full-game NE points ==="
            << std::endl;
  {
    // Single shared trunk policy (round < 3) used across all stitched variants.
    auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
    std::unordered_set<std::string> trunk_all_infosets;
    TabularPolicy trunk_policy = uniform_base;
    for (const auto& [player, info_states] : trunk_infostates) {
      for (const auto& is : info_states) {
        trunk_all_infosets.insert(is);
        auto ap = full_ne.GetStatePolicy(is);
        SPIEL_CHECK_FALSE(ap.empty());
        trunk_policy.SetStatePolicy(is, ap);
      }
    }

    std::unordered_set<std::string> stitchable_p0;
    std::unordered_set<std::string> stitchable_p1;
    for (const auto& g : groups) {
      auto it = decomp.grouped_subgames.find(g);
      if (it == decomp.grouped_subgames.end()) continue;
      auto info = CollectSubgameInfoStatesPerPlayer(it->second);
      stitchable_p0.insert(info[0].begin(), info[0].end());
      stitchable_p1.insert(info[1].begin(), info[1].end());
    }

    auto choose_pivot_infoset = [&](int player,
                                    const std::unordered_set<std::string>& allowed) {
      algorithms::ortools::SequenceFormLpSpecification spec(*game, "CLP", false);
      spec.SpecifyLinearProgram(player);
      spec.Solve();
      TabularPolicy p = spec.OptimalPolicy(player, true);
      std::string best_is;
      double best_mix = -1.0;
      for (const auto& kv : p.PolicyTable()) {
        const std::string& is = kv.first;
        if (allowed.count(is) == 0) continue;
        const ActionsAndProbs& ap = kv.second;
        double maxp = 0.0, minp = 1.0;
        for (const auto& a : ap) {
          maxp = std::max(maxp, a.second);
          minp = std::min(minp, a.second);
        }
        // Prefer genuinely mixed infosets.
        if (maxp < 0.999 && minp > 1e-6 && (1.0 - maxp) > best_mix) {
          best_mix = 1.0 - maxp;
          best_is = is;
        }
      }
      return best_is;
    };

    auto make_policy_with_reach_constraint = [&](int player,
                                                 const std::string& pivot_is,
                                                 bool increase_reach,
                                                 double value_floor,
                                                 TabularPolicy fallback) {
      std::vector<double> deltas = {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6};
      for (double delta : deltas) {
        algorithms::ortools::SequenceFormLpSpecification spec(*game, "CLP", true);
        spec.SpecifyLinearProgram(player);
        double base_value = spec.Solve();
        if (std::isnan(base_value)) continue;

        auto* node =
            spec.trees()[player]->DecisionNodeFromInfostateString(pivot_is);
        if (node == nullptr) continue;
        auto ns_it = spec.node_spec().find(node);
        if (ns_it == spec.node_spec().end()) continue;
        auto ns = ns_it->second;
        if (ns.var_reach_prob == nullptr) continue;
        const double base_reach = ns.var_reach_prob->solution_value();

        auto* solver = spec.solver();
        auto root_node = spec.roots()[player];
        auto root_cf = spec.node_spec().at(root_node).var_cf_value;
        auto* keep_value = solver->MakeRowConstraint(value_floor, 1e9);
        keep_value->SetCoefficient(root_cf, 1.0);

        if (increase_reach) {
          auto* c = solver->MakeRowConstraint(base_reach + delta, 1e9);
          c->SetCoefficient(ns.var_reach_prob, 1.0);
        } else {
          auto* c = solver->MakeRowConstraint(-1e9, base_reach - delta);
          c->SetCoefficient(ns.var_reach_prob, 1.0);
        }

        double constrained_value = spec.Solve();
        if (!std::isnan(constrained_value)) {
          return spec.OptimalPolicy(player, true);
        }
      }
      return fallback;
    };

    // Baseline full NE value.
    algorithms::ortools::SequenceFormLpSpecification spec0(*game, "CLP", false);
    spec0.SpecifyLinearProgram(0);
    double v0 = spec0.Solve();
    algorithms::ortools::SequenceFormLpSpecification spec1(*game, "CLP", false);
    spec1.SpecifyLinearProgram(1);
    double v1 = spec1.Solve();
    std::cout << "  Baseline full LP values P0/P1 = " << v0 << " / " << v1
              << std::endl;

    // Two alternative optimal points per player, selected by explicit LP
    // constraints on a mixed infoset realization weight.
    const std::string pivot0 = choose_pivot_infoset(0, stitchable_p0);
    SPIEL_CHECK_FALSE(pivot0.empty());
    const std::string pivot1 = choose_pivot_infoset(1, stitchable_p1);
    SPIEL_CHECK_FALSE(pivot1.empty());
    std::cout << "  pivot infosets: P0='" << pivot0 << "' P1='" << pivot1 << "'"
              << std::endl;

    algorithms::ortools::SequenceFormLpSpecification b0(*game, "CLP", false);
    b0.SpecifyLinearProgram(0);
    b0.Solve();
    TabularPolicy p0_base = b0.OptimalPolicy(0, true);
    algorithms::ortools::SequenceFormLpSpecification b1(*game, "CLP", false);
    b1.SpecifyLinearProgram(1);
    b1.Solve();
    TabularPolicy p1_base = b1.OptimalPolicy(1, true);

    TabularPolicy p0_a =
        make_policy_with_reach_constraint(0, pivot0, true, v0 - 1e-10, p0_base);
    TabularPolicy p0_b =
        make_policy_with_reach_constraint(0, pivot0, false, v0 - 1e-10, p0_base);
    TabularPolicy p1_a =
        make_policy_with_reach_constraint(1, pivot1, true, v1 - 1e-10, p1_base);
    TabularPolicy p1_b =
        make_policy_with_reach_constraint(1, pivot1, false, v1 - 1e-10, p1_base);

    // Sanity: exploitability of raw A/B combinations before any group-wise
    // stitching. If these are non-zero, the source solve itself is not an NE.
    auto all_infosets = CollectAllInfoStates(*game);
    auto make_full_player_policy = [&](int player, const TabularPolicy& solve_policy) {
      TabularPolicy out = full_ne;
      for (const auto& is : all_infosets[player]) {
        auto ap = solve_policy.GetStatePolicy(is);
        if (!ap.empty()) out.SetStatePolicy(is, ap);
      }
      return out;
    };
    TabularPolicy p0_a_full = make_full_player_policy(0, p0_a);
    TabularPolicy p0_b_full = make_full_player_policy(0, p0_b);
    TabularPolicy p1_a_full = make_full_player_policy(1, p1_a);
    TabularPolicy p1_b_full = make_full_player_policy(1, p1_b);

    TabularPolicy raw_a = full_ne;
    for (const auto& is : all_infosets[0]) raw_a.SetStatePolicy(is, p0_a_full.GetStatePolicy(is));
    for (const auto& is : all_infosets[1]) raw_a.SetStatePolicy(is, p1_a_full.GetStatePolicy(is));
    TabularPolicy raw_b = full_ne;
    for (const auto& is : all_infosets[0]) raw_b.SetStatePolicy(is, p0_b_full.GetStatePolicy(is));
    for (const auto& is : all_infosets[1]) raw_b.SetStatePolicy(is, p1_b_full.GetStatePolicy(is));
    double exp_raw_a = algorithms::Exploitability(*game, raw_a);
    double exp_raw_b = algorithms::Exploitability(*game, raw_b);
    std::cout << "  raw-A (pre-stitch) exp = " << exp_raw_a << std::endl;
    std::cout << "  raw-B (pre-stitch) exp = " << exp_raw_b << std::endl;

    std::unordered_map<std::string, int> group_index;
    for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
      group_index[groups[gi]] = gi;
    }
    std::unordered_map<std::string, int> infoset_group[2];
    for (const auto& g : groups) {
      auto it = decomp.grouped_subgames.find(g);
      if (it == decomp.grouped_subgames.end()) continue;
      auto info = CollectSubgameInfoStatesPerPlayer(it->second);
      for (const auto& is : info[0]) infoset_group[0][is] = group_index[g];
      for (const auto& is : info[1]) infoset_group[1][is] = group_index[g];
    }

    auto stitch_whole_player = [&](const TabularPolicy& a_full,
                                   const TabularPolicy& b_full,
                                   int pl, bool all_a_mode, bool all_b_mode,
                                   TabularPolicy* out) {
      for (const auto& is : all_infosets[pl]) {
        bool use_a = true;
        if (all_a_mode) {
          use_a = true;
        } else if (all_b_mode) {
          use_a = false;
        } else {
          auto itg = infoset_group[pl].find(is);
          if (itg != infoset_group[pl].end()) {
            use_a = (itg->second % 2 == 0);
          } else {
            // Outside decomposed subgames (e.g., trunk): choose A consistently.
            use_a = true;
          }
        }
        const ActionsAndProbs ap =
            use_a ? a_full.GetStatePolicy(is) : b_full.GetStatePolicy(is);
        SPIEL_CHECK_FALSE(ap.empty());
        out->SetStatePolicy(is, ap);
      }
    };

    // Controls.
    TabularPolicy all_a = full_ne;
    stitch_whole_player(p0_a_full, p0_b_full, 0, /*all_a_mode=*/true,
                        /*all_b_mode=*/false, &all_a);
    stitch_whole_player(p1_a_full, p1_b_full, 1, /*all_a_mode=*/true,
                        /*all_b_mode=*/false, &all_a);
    // Sanity: all-A stitch should reconstruct raw-A if coverage/mapping is exact.
    auto policy_l1_diff = [&](const TabularPolicy& x, const TabularPolicy& y) {
      std::unordered_set<std::string> keys;
      for (const auto& kv : x.PolicyTable()) keys.insert(kv.first);
      for (const auto& kv : y.PolicyTable()) keys.insert(kv.first);
      int num_changed_infosets = 0;
      double total_l1 = 0.0;
      int printed = 0;
      for (const auto& is : keys) {
        ActionsAndProbs ax = x.GetStatePolicy(is);
        ActionsAndProbs ay = y.GetStatePolicy(is);
        std::unordered_map<Action, double> mx, my;
        for (const auto& [a, p] : ax) mx[a] = p;
        for (const auto& [a, p] : ay) my[a] = p;
        std::unordered_set<Action> acts;
        for (const auto& [a, p] : mx) acts.insert(a);
        for (const auto& [a, p] : my) acts.insert(a);
        double l1 = 0.0;
        for (Action a : acts) l1 += std::abs(mx[a] - my[a]);
        if (l1 > 1e-12) {
          ++num_changed_infosets;
          total_l1 += l1;
          if (printed < 8) {
            ++printed;
            std::cout << "    diff[" << printed << "] " << is
                      << " l1=" << l1 << std::endl;
          }
        }
      }
      return std::make_pair(num_changed_infosets, total_l1);
    };
    auto [nchg_a, l1_a] = policy_l1_diff(raw_a, all_a);
    std::cout << "  raw-A vs all-A diffs: infosets=" << nchg_a
              << " total_l1=" << l1_a << std::endl;
    double exp_all_a = algorithms::Exploitability(*game, all_a);
    std::cout << "  all-A stitched exp = " << exp_all_a << std::endl;

    TabularPolicy all_b = full_ne;
    stitch_whole_player(p0_a_full, p0_b_full, 0, /*all_a_mode=*/false,
                        /*all_b_mode=*/true, &all_b);
    stitch_whole_player(p1_a_full, p1_b_full, 1, /*all_a_mode=*/false,
                        /*all_b_mode=*/true, &all_b);
    double exp_all_b = algorithms::Exploitability(*game, all_b);
    std::cout << "  all-B stitched exp = " << exp_all_b << std::endl;

    // Stitch by whole public-state groups: alternate A/B per group.
    TabularPolicy mixed = full_ne;
    stitch_whole_player(p0_a_full, p0_b_full, 0, /*all_a_mode=*/false,
                        /*all_b_mode=*/false, &mixed);
    stitch_whole_player(p1_a_full, p1_b_full, 1, /*all_a_mode=*/false,
                        /*all_b_mode=*/false, &mixed);
    double exp_mixed = algorithms::Exploitability(*game, mixed);
    std::cout << "  mixed-group stitched exp = " << exp_mixed << std::endl;
  }

  std::cout << "RunLeducHalfSplitExperiment DONE" << std::endl;
}

void RunStitchingOnlyTestR() {
  std::cout << "RunStitchingOnlyTestR..." << std::endl;
  auto game = LoadGame("leduc_poker");
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto groups = SortedGroups(decomp);
  auto [full_ne, full_val] = algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  double full_ne_exp = algorithms::Exploitability(*game, full_ne);
  std::cout << "  Full LP NE exp: " << full_ne_exp << std::endl;
  TabularPolicy uniform_base(*game);
  auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::unordered_set<std::string> trunk_all_infosets;
  TabularPolicy trunk_policy = uniform_base;
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& is : info_states) {
      trunk_all_infosets.insert(is);
      auto ap = full_ne.GetStatePolicy(is);
      SPIEL_CHECK_FALSE(ap.empty());
      trunk_policy.SetStatePolicy(is, ap);
    }
  }
  std::cout << "\n=== [Test R] Stitching-only with different full-game NE points ==="
            << std::endl;
  {
    auto choose_pivot_infoset = [&](int player) {
      algorithms::ortools::SequenceFormLpSpecification spec(*game, "CLP", false);
      spec.SpecifyLinearProgram(player);
      spec.Solve();
      TabularPolicy p = spec.OptimalPolicy(player, true);
      std::string best_is;
      double best_mix = -1.0;
      for (const auto& kv : p.PolicyTable()) {
        const std::string& is = kv.first;
        const ActionsAndProbs& ap = kv.second;
        double maxp = 0.0, minp = 1.0;
        for (const auto& a : ap) {
          maxp = std::max(maxp, a.second);
          minp = std::min(minp, a.second);
        }
        if (maxp < 0.999 && minp > 1e-6 && (1.0 - maxp) > best_mix) {
          best_mix = 1.0 - maxp;
          best_is = is;
        }
      }
      return best_is;
    };

    auto make_policy_with_reach_constraint = [&](int player,
                                                 const std::string& pivot_is,
                                                 bool increase_reach,
                                                 double value_floor,
                                                 TabularPolicy fallback) {
      std::vector<double> deltas = {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6};
      for (double delta : deltas) {
        algorithms::ortools::SequenceFormLpSpecification spec(*game, "CLP", true);
        spec.SpecifyLinearProgram(player);
        double base_value = spec.Solve();
        if (std::isnan(base_value)) continue;

        auto* node = spec.trees()[player]->DecisionNodeFromInfostateString(pivot_is);
        if (node == nullptr) continue;
        auto ns_it = spec.node_spec().find(node);
        if (ns_it == spec.node_spec().end()) continue;
        auto ns = ns_it->second;
        if (ns.var_reach_prob == nullptr) continue;
        const double base_reach = ns.var_reach_prob->solution_value();

        auto* solver = spec.solver();
        auto root_node = spec.roots()[player];
        auto root_cf = spec.node_spec().at(root_node).var_cf_value;
        auto* keep_value = solver->MakeRowConstraint(value_floor, 1e9);
        keep_value->SetCoefficient(root_cf, 1.0);

        if (increase_reach) {
          auto* c = solver->MakeRowConstraint(base_reach + delta, 1e9);
          c->SetCoefficient(ns.var_reach_prob, 1.0);
        } else {
          auto* c = solver->MakeRowConstraint(-1e9, base_reach - delta);
          c->SetCoefficient(ns.var_reach_prob, 1.0);
        }

        double constrained_value = spec.Solve();
        if (!std::isnan(constrained_value)) {
          return spec.OptimalPolicy(player, true);
        }
      }
      return fallback;
    };

    algorithms::ortools::SequenceFormLpSpecification spec0(*game, "CLP", false);
    spec0.SpecifyLinearProgram(0);
    double v0 = spec0.Solve();
    algorithms::ortools::SequenceFormLpSpecification spec1(*game, "CLP", false);
    spec1.SpecifyLinearProgram(1);
    double v1 = spec1.Solve();
    std::cout << "  Baseline full LP values P0/P1 = " << v0 << " / " << v1
              << std::endl;

    const std::string pivot0 = choose_pivot_infoset(0);
    SPIEL_CHECK_FALSE(pivot0.empty());
    const std::string pivot1 = choose_pivot_infoset(1);
    SPIEL_CHECK_FALSE(pivot1.empty());
    std::cout << "  pivot infosets: P0='" << pivot0 << "' P1='" << pivot1 << "'"
              << std::endl;

    algorithms::ortools::SequenceFormLpSpecification b0(*game, "CLP", false);
    b0.SpecifyLinearProgram(0);
    b0.Solve();
    TabularPolicy p0_base = b0.OptimalPolicy(0, true);
    algorithms::ortools::SequenceFormLpSpecification b1(*game, "CLP", false);
    b1.SpecifyLinearProgram(1);
    b1.Solve();
    TabularPolicy p1_base = b1.OptimalPolicy(1, true);

    TabularPolicy p0_a =
        make_policy_with_reach_constraint(0, pivot0, true, v0 - 1e-10, p0_base);
    TabularPolicy p0_b =
        make_policy_with_reach_constraint(0, pivot0, false, v0 - 1e-10, p0_base);
    TabularPolicy p1_a =
        make_policy_with_reach_constraint(1, pivot1, true, v1 - 1e-10, p1_base);
    TabularPolicy p1_b =
        make_policy_with_reach_constraint(1, pivot1, false, v1 - 1e-10, p1_base);

    auto all_infosets = CollectAllInfoStates(*game);
    auto make_full_player_policy = [&](int player, const TabularPolicy& solve_policy) {
      TabularPolicy out = uniform_base;
      for (const auto& is : all_infosets[player]) {
        auto ap = solve_policy.GetStatePolicy(is);
        if (!ap.empty()) out.SetStatePolicy(is, ap);
      }
      // Enforce single shared trunk policy.
      for (const auto& is : all_infosets[player]) {
        if (trunk_all_infosets.count(is) == 0) continue;
        auto ap = trunk_policy.GetStatePolicy(is);
        SPIEL_CHECK_FALSE(ap.empty());
        out.SetStatePolicy(is, ap);
      }
      return out;
    };
    TabularPolicy p0_a_full = make_full_player_policy(0, p0_a);
    TabularPolicy p0_b_full = make_full_player_policy(0, p0_b);
    TabularPolicy p1_a_full = make_full_player_policy(1, p1_a);
    TabularPolicy p1_b_full = make_full_player_policy(1, p1_b);

    TabularPolicy raw_a = uniform_base;
    for (const auto& is : all_infosets[0]) raw_a.SetStatePolicy(is, p0_a_full.GetStatePolicy(is));
    for (const auto& is : all_infosets[1]) raw_a.SetStatePolicy(is, p1_a_full.GetStatePolicy(is));
    TabularPolicy raw_b = uniform_base;
    for (const auto& is : all_infosets[0]) raw_b.SetStatePolicy(is, p0_b_full.GetStatePolicy(is));
    for (const auto& is : all_infosets[1]) raw_b.SetStatePolicy(is, p1_b_full.GetStatePolicy(is));
    double exp_raw_a = algorithms::Exploitability(*game, raw_a);
    double exp_raw_b = algorithms::Exploitability(*game, raw_b);
    std::cout << "  raw-A (pre-stitch) exp = " << exp_raw_a << std::endl;
    std::cout << "  raw-B (pre-stitch) exp = " << exp_raw_b << std::endl;

    std::unordered_map<std::string, int> group_index;
    for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
      group_index[groups[gi]] = gi;
    }
    std::unordered_map<std::string, int> infoset_group[2];
    for (const auto& g : groups) {
      auto it = decomp.grouped_subgames.find(g);
      if (it == decomp.grouped_subgames.end()) continue;
      auto info = CollectSubgameInfoStatesPerPlayer(it->second);
      for (const auto& is : info[0]) infoset_group[0][is] = group_index[g];
      for (const auto& is : info[1]) infoset_group[1][is] = group_index[g];
    }

    auto stitch_whole_player = [&](const TabularPolicy& a_full,
                                   const TabularPolicy& b_full,
                                   int pl, bool all_a_mode, bool all_b_mode,
                                   TabularPolicy* out) {
      for (const auto& is : all_infosets[pl]) {
        bool use_a = true;
        if (all_a_mode) {
          use_a = true;
        } else if (all_b_mode) {
          use_a = false;
        } else {
          auto itg = infoset_group[pl].find(is);
          if (itg != infoset_group[pl].end()) {
            use_a = (itg->second % 2 == 0);
          } else {
            if (trunk_all_infosets.count(is) == 0) {
              SpielFatalError(absl::StrCat(
                  "Mixed stitching missing infoset->group mapping for P", pl,
                  " infoset: ", is));
            }
            use_a = true;
          }
        }
        const ActionsAndProbs ap =
            use_a ? a_full.GetStatePolicy(is) : b_full.GetStatePolicy(is);
        SPIEL_CHECK_FALSE(ap.empty());
        out->SetStatePolicy(is, ap);
      }
    };

    TabularPolicy all_a = uniform_base;
    stitch_whole_player(p0_a_full, p0_b_full, 0, /*all_a_mode=*/true,
                        /*all_b_mode=*/false, &all_a);
    stitch_whole_player(p1_a_full, p1_b_full, 1, /*all_a_mode=*/true,
                        /*all_b_mode=*/false, &all_a);
    auto policy_l1_diff = [&](const TabularPolicy& x, const TabularPolicy& y) {
      std::unordered_set<std::string> keys;
      for (const auto& kv : x.PolicyTable()) keys.insert(kv.first);
      for (const auto& kv : y.PolicyTable()) keys.insert(kv.first);
      int num_changed_infosets = 0;
      double total_l1 = 0.0;
      for (const auto& is : keys) {
        ActionsAndProbs ax = x.GetStatePolicy(is);
        ActionsAndProbs ay = y.GetStatePolicy(is);
        std::unordered_map<Action, double> mx, my;
        for (const auto& [a, p] : ax) mx[a] = p;
        for (const auto& [a, p] : ay) my[a] = p;
        std::unordered_set<Action> acts;
        for (const auto& [a, p] : mx) acts.insert(a);
        for (const auto& [a, p] : my) acts.insert(a);
        double l1 = 0.0;
        for (Action a : acts) l1 += std::abs(mx[a] - my[a]);
        if (l1 > 1e-12) {
          ++num_changed_infosets;
          total_l1 += l1;
        }
      }
      return std::make_pair(num_changed_infosets, total_l1);
    };
    auto [nchg_a, l1_a] = policy_l1_diff(raw_a, all_a);
    std::cout << "  raw-A vs all-A diffs: infosets=" << nchg_a
              << " total_l1=" << l1_a << std::endl;
    double exp_all_a = algorithms::Exploitability(*game, all_a);
    std::cout << "  all-A stitched exp = " << exp_all_a << std::endl;

    TabularPolicy all_b = uniform_base;
    stitch_whole_player(p0_a_full, p0_b_full, 0, /*all_a_mode=*/false,
                        /*all_b_mode=*/true, &all_b);
    stitch_whole_player(p1_a_full, p1_b_full, 1, /*all_a_mode=*/false,
                        /*all_b_mode=*/true, &all_b);
    double exp_all_b = algorithms::Exploitability(*game, all_b);
    std::cout << "  all-B stitched exp = " << exp_all_b << std::endl;

    // Cross-player stitching from the four player-specific solves.
    auto build_profile = [&](const TabularPolicy& p0_src,
                             const TabularPolicy& p1_src) {
      TabularPolicy out = uniform_base;
      for (const auto& is : all_infosets[0]) {
        out.SetStatePolicy(is, p0_src.GetStatePolicy(is));
      }
      for (const auto& is : all_infosets[1]) {
        out.SetStatePolicy(is, p1_src.GetStatePolicy(is));
      }
      return out;
    };
    TabularPolicy prof_aa = build_profile(p0_a_full, p1_a_full);
    TabularPolicy prof_ab = build_profile(p0_a_full, p1_b_full);
    TabularPolicy prof_ba = build_profile(p0_b_full, p1_a_full);
    TabularPolicy prof_bb = build_profile(p0_b_full, p1_b_full);
    std::cout << "  profile P0A+P1A exp = "
              << algorithms::Exploitability(*game, prof_aa) << std::endl;
    std::cout << "  profile P0A+P1B exp = "
              << algorithms::Exploitability(*game, prof_ab) << std::endl;
    std::cout << "  profile P0B+P1A exp = "
              << algorithms::Exploitability(*game, prof_ba) << std::endl;
    std::cout << "  profile P0B+P1B exp = "
              << algorithms::Exploitability(*game, prof_bb) << std::endl;

    // Public-state-consistent profile stitching: each public group is chosen as
    // whole profile A (P0A+P1A) or whole profile B (P0B+P1B).
    auto stitch_profile_by_groups = [&](const std::unordered_map<int, bool>& use_a_group) {
      TabularPolicy out = uniform_base;
      for (int pl = 0; pl < 2; ++pl) {
        for (const auto& is : all_infosets[pl]) {
          if (trunk_all_infosets.count(is) > 0) {
            out.SetStatePolicy(is, trunk_policy.GetStatePolicy(is));
            continue;
          }
          auto itg = infoset_group[pl].find(is);
          if (itg == infoset_group[pl].end()) {
            SpielFatalError(absl::StrCat(
                "Profile group stitching missing infoset->group mapping for P",
                pl, " infoset: ", is));
          }
          const int gid = itg->second;
          const bool use_a = use_a_group.at(gid);
          const ActionsAndProbs ap =
              (pl == 0)
                  ? (use_a ? p0_a_full.GetStatePolicy(is)
                           : p0_b_full.GetStatePolicy(is))
                  : (use_a ? p1_a_full.GetStatePolicy(is)
                           : p1_b_full.GetStatePolicy(is));
          SPIEL_CHECK_FALSE(ap.empty());
          out.SetStatePolicy(is, ap);
        }
      }
      return out;
    };

    std::unordered_map<int, bool> pattern_parity;
    std::unordered_map<int, bool> pattern_first_two_thirds_a;
    std::unordered_map<int, bool> pattern_hash;
    for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
      pattern_parity[gi] = (gi % 2 == 0);
      pattern_first_two_thirds_a[gi] = (gi < (2 * static_cast<int>(groups.size()) / 3));
      pattern_hash[gi] = ((gi * 17 + 5) % 7 < 4);
    }
    TabularPolicy mix_parity = stitch_profile_by_groups(pattern_parity);
    TabularPolicy mix_two_thirds = stitch_profile_by_groups(pattern_first_two_thirds_a);
    TabularPolicy mix_hash = stitch_profile_by_groups(pattern_hash);
    std::cout << "  profile-group mixed (parity) exp = "
              << algorithms::Exploitability(*game, mix_parity) << std::endl;
    std::cout << "  profile-group mixed (first-2/3 A) exp = "
              << algorithms::Exploitability(*game, mix_two_thirds) << std::endl;
    std::cout << "  profile-group mixed (hash) exp = "
              << algorithms::Exploitability(*game, mix_hash) << std::endl;
  }
  std::cout << "RunStitchingOnlyTestR DONE" << std::endl;
}

void RunStitchingOnlyTestRStrict() {
  std::cout << "RunStitchingOnlyTestRStrict..." << std::endl;
  auto game = LoadGame("leduc_poker");
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto groups = SortedGroups(decomp);
  TabularPolicy uniform_base(*game);

  // Step 1: THE NE.
  auto [full_ne, full_val] = algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  std::cout << "  THE_NE exploitability = "
            << algorithms::Exploitability(*game, full_ne) << std::endl;

  // Shared trunk policy from THE_NE.
  auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::unordered_set<std::string> trunk_all_infosets;
  TabularPolicy trunk_policy = uniform_base;
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& is : info_states) {
      trunk_all_infosets.insert(is);
      trunk_policy.SetStatePolicy(is, full_ne.GetStatePolicy(is));
    }
  }

  auto all_infosets = CollectAllInfoStates(*game);
  std::unordered_map<std::string, int> group_index;
  std::unordered_map<std::string, int> infoset_group[2];
  std::unordered_set<std::string> allowed_all[2];
  for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
    group_index[groups[gi]] = gi;
  }
  for (const auto& g : groups) {
    auto it = decomp.grouped_subgames.find(g);
    if (it == decomp.grouped_subgames.end()) continue;
    auto info = CollectSubgameInfoStatesPerPlayer(it->second);
    for (const auto& is : info[0]) {
      infoset_group[0][is] = group_index[g];
      allowed_all[0].insert(is);
    }
    for (const auto& is : info[1]) {
      infoset_group[1][is] = group_index[g];
      allowed_all[1].insert(is);
    }
  }

  auto make_all_target_fg = [&](int rp) {
    const std::string kAll = "__all_tgt__";
    std::unordered_map<std::string, std::vector<std::string>> bbg;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots) bbg[kAll].push_back(root->HistoryString());
    }
    auto trunk_ptr = std::make_shared<TabularPolicy>(trunk_policy);
    return CreateFullGadgetGame(game, trunk_ptr, rp, kAll, bbg,
                                FullGadgetGame::Mode::kTrunk,
                                {}, {}, /*enumerate_boundary_portfolios=*/false);
  };

  auto solve_variant_from_locked_fg = [&](int rp, bool want_higher) {
    auto fg = make_all_target_fg(rp);
    algorithms::ortools::SequenceFormLpSpecification base_spec(*fg, "CLP", false);
    base_spec.SpecifyLinearProgram(rp);
    const double v = base_spec.Solve();
    TabularPolicy pol_base = base_spec.OptimalPolicy(rp, true);

    std::string pivot;
    for (const auto& kv : pol_base.PolicyTable()) {
      double maxp = 0.0, minp = 1.0;
      for (const auto& ap : kv.second) {
        maxp = std::max(maxp, ap.second);
        minp = std::min(minp, ap.second);
      }
      if (maxp < 0.999 && minp > 1e-6) {
        pivot = kv.first;
        break;
      }
    }

    TabularPolicy chosen_fg = pol_base;
    if (!pivot.empty()) {
      // Force A/B apart by optimizing pivot realization with a tiny value slack.
      std::vector<double> slacks = {1e-10, 1e-8, 1e-7, 1e-6, 1e-5};
      for (double slack : slacks) {
        algorithms::ortools::SequenceFormLpSpecification spec(*fg, "CLP", true);
        spec.SpecifyLinearProgram(rp);
        double vb = spec.Solve();
        if (std::isnan(vb)) continue;
        auto* node = spec.trees()[rp]->DecisionNodeFromInfostateString(pivot);
        if (node == nullptr) continue;
        auto ns_it = spec.node_spec().find(node);
        if (ns_it == spec.node_spec().end()) continue;
        auto ns = ns_it->second;
        if (ns.var_reach_prob == nullptr) continue;

        auto* solver = spec.solver();
        auto root_cf = spec.node_spec().at(spec.roots()[rp]).var_cf_value;
        auto* keep_value = solver->MakeRowConstraint(v - slack, 1e9);
        keep_value->SetCoefficient(root_cf, 1.0);

        auto* obj = solver->MutableObjective();
        for (operations_research::MPVariable* var : solver->variables()) {
          obj->SetCoefficient(var, 0.0);
        }
        obj->SetCoefficient(ns.var_reach_prob, want_higher ? 1.0 : -1.0);
        obj->SetMaximization();

        double vc = spec.Solve();
        if (!std::isnan(vc)) {
          chosen_fg = spec.OptimalPolicy(rp, true);
          break;
        }
      }
    }

    TabularPolicy out = uniform_base;
    for (const auto& is : all_infosets[rp]) {
      if (trunk_all_infosets.count(is)) {
        out.SetStatePolicy(is, trunk_policy.GetStatePolicy(is));
      } else {
        out.SetStatePolicy(is, full_ne.GetStatePolicy(is));
      }
    }
    StitchPlayerFromFGPolicy(chosen_fg, rp, allowed_all[rp],
                             trunk_all_infosets, &out);
    return out;
  };

  // Step 2: lock P0 trunk and solve twice (A/B); Step 3 similarly for P1.
  TabularPolicy p0_a = solve_variant_from_locked_fg(0, /*want_higher=*/true);
  TabularPolicy p0_b = solve_variant_from_locked_fg(0, /*want_higher=*/false);
  TabularPolicy p1_a = solve_variant_from_locked_fg(1, /*want_higher=*/true);
  TabularPolicy p1_b = solve_variant_from_locked_fg(1, /*want_higher=*/false);

  auto player_policy_l1 = [&](int pl, const TabularPolicy& a, const TabularPolicy& b) {
    double tot = 0.0;
    int changed = 0;
    for (const auto& is : all_infosets[pl]) {
      auto pa = a.GetStatePolicy(is);
      auto pb = b.GetStatePolicy(is);
      if (pa.empty() || pb.empty()) continue;
      double d = L1PolicyDelta(pa, pb);
      if (d > 1e-12) {
        ++changed;
        tot += d;
      }
    }
    std::cout << "  P" << pl << " A-vs-B policy diff: changed=" << changed
              << " total_l1=" << tot << std::endl;
    return std::make_pair(changed, tot);
  };
  auto make_tiny_perturbed_from_ne = [&](int pl, bool up) {
    TabularPolicy out = uniform_base;
    for (const auto& is : all_infosets[pl]) {
      if (trunk_all_infosets.count(is)) {
        out.SetStatePolicy(is, trunk_policy.GetStatePolicy(is));
      } else {
        out.SetStatePolicy(is, full_ne.GetStatePolicy(is));
      }
    }
    const double eps = 1e-5;
    for (const auto& is : all_infosets[pl]) {
      if (trunk_all_infosets.count(is)) continue;
      ActionsAndProbs ap = out.GetStatePolicy(is);
      if (ap.size() < 2) continue;
      int i0 = -1, i1 = -1;
      for (int i = 0; i < static_cast<int>(ap.size()); ++i) {
        if (ap[i].second > 1e-6 && ap[i].second < 1.0 - 1e-6) {
          if (i0 == -1) i0 = i;
          else if (i1 == -1) {
            i1 = i;
            break;
          }
        }
      }
      if (i0 == -1 || i1 == -1) continue;
      if (up) {
        double d = std::min(eps, ap[i1].second * 0.5);
        ap[i0].second += d;
        ap[i1].second -= d;
      } else {
        double d = std::min(eps, ap[i0].second * 0.5);
        ap[i0].second -= d;
        ap[i1].second += d;
      }
      out.SetStatePolicy(is, ap);
      return out;
    }
    return out;
  };

  auto [p0_changed, p0_l1] = player_policy_l1(0, p0_a, p0_b);
  auto [p1_changed, p1_l1] = player_policy_l1(1, p1_a, p1_b);
  if (p0_changed == 0) {
    p0_a = make_tiny_perturbed_from_ne(0, /*up=*/true);
    p0_b = make_tiny_perturbed_from_ne(0, /*up=*/false);
    std::cout << "  P0 fallback: tiny NE perturbation applied." << std::endl;
  }
  if (p1_changed == 0) {
    p1_a = make_tiny_perturbed_from_ne(1, /*up=*/true);
    p1_b = make_tiny_perturbed_from_ne(1, /*up=*/false);
    std::cout << "  P1 fallback: tiny NE perturbation applied." << std::endl;
  }
  player_policy_l1(0, p0_a, p0_b);
  player_policy_l1(1, p1_a, p1_b);

  auto build_profile = [&](const TabularPolicy& p0_src, const TabularPolicy& p1_src) {
    TabularPolicy out = uniform_base;
    for (const auto& is : all_infosets[0]) out.SetStatePolicy(is, p0_src.GetStatePolicy(is));
    for (const auto& is : all_infosets[1]) out.SetStatePolicy(is, p1_src.GetStatePolicy(is));
    return out;
  };

  TabularPolicy prof_aa = build_profile(p0_a, p1_a);
  TabularPolicy prof_ab = build_profile(p0_a, p1_b);
  TabularPolicy prof_ba = build_profile(p0_b, p1_a);
  TabularPolicy prof_bb = build_profile(p0_b, p1_b);
  std::cout << "  profile P0A+P1A exp = "
            << algorithms::Exploitability(*game, prof_aa) << std::endl;
  std::cout << "  profile P0A+P1B exp = "
            << algorithms::Exploitability(*game, prof_ab) << std::endl;
  std::cout << "  profile P0B+P1A exp = "
            << algorithms::Exploitability(*game, prof_ba) << std::endl;
  std::cout << "  profile P0B+P1B exp = "
            << algorithms::Exploitability(*game, prof_bb) << std::endl;

  auto stitch_profile_by_groups = [&](const std::unordered_map<int, bool>& use_a_group) {
    TabularPolicy out = uniform_base;
    for (int pl = 0; pl < 2; ++pl) {
      for (const auto& is : all_infosets[pl]) {
        if (trunk_all_infosets.count(is)) {
          out.SetStatePolicy(is, trunk_policy.GetStatePolicy(is));
          continue;
        }
        auto itg = infoset_group[pl].find(is);
        if (itg == infoset_group[pl].end()) {
          SpielFatalError(absl::StrCat(
              "Profile group stitching missing infoset->group mapping for P",
              pl, " infoset: ", is));
        }
        const bool use_a = use_a_group.at(itg->second);
        if (pl == 0) {
          out.SetStatePolicy(is, use_a ? p0_a.GetStatePolicy(is)
                                       : p0_b.GetStatePolicy(is));
        } else {
          out.SetStatePolicy(is, use_a ? p1_a.GetStatePolicy(is)
                                       : p1_b.GetStatePolicy(is));
        }
      }
    }
    return out;
  };

  std::unordered_map<int, bool> pattern_parity;
  std::unordered_map<int, bool> pattern_first_two_thirds_a;
  std::unordered_map<int, bool> pattern_hash;
  for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
    pattern_parity[gi] = (gi % 2 == 0);
    pattern_first_two_thirds_a[gi] = (gi < (2 * static_cast<int>(groups.size()) / 3));
    pattern_hash[gi] = ((gi * 17 + 5) % 7 < 4);
  }
  std::cout << "  profile-group mixed (parity) exp = "
            << algorithms::Exploitability(*game, stitch_profile_by_groups(pattern_parity))
            << std::endl;
  std::cout << "  profile-group mixed (first-2/3 A) exp = "
            << algorithms::Exploitability(*game, stitch_profile_by_groups(pattern_first_two_thirds_a))
            << std::endl;
  std::cout << "  profile-group mixed (hash) exp = "
            << algorithms::Exploitability(*game, stitch_profile_by_groups(pattern_hash))
            << std::endl;
  std::cout << "RunStitchingOnlyTestRStrict DONE" << std::endl;
}

void RunClassicSingleTargetStitchRecheck() {
  std::cout << "RunClassicSingleTargetStitchRecheck..." << std::endl;
  auto game = LoadGame("leduc_poker");
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto groups = SortedGroups(decomp);
  auto [full_ne, full_val] = algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  std::cout << "  THE_NE exploitability = "
            << algorithms::Exploitability(*game, full_ne) << std::endl;

  TabularPolicy uniform_base(*game);
  auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::unordered_set<std::string> trunk_all_infosets;
  TabularPolicy trunk_policy = uniform_base;
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& is : info_states) {
      trunk_all_infosets.insert(is);
      trunk_policy.SetStatePolicy(is, full_ne.GetStatePolicy(is));
    }
  }

  auto all_infosets = CollectAllInfoStates(*game);
  std::unordered_map<std::string, std::unordered_set<std::string>> allowed_by_group[2];
  std::unordered_map<std::string, std::string> owner_group[2];
  for (const auto& g : groups) {
    auto it = decomp.grouped_subgames.find(g);
    if (it == decomp.grouped_subgames.end()) continue;
    auto info = CollectSubgameInfoStatesPerPlayer(it->second);
    allowed_by_group[0][g].insert(info[0].begin(), info[0].end());
    allowed_by_group[1][g].insert(info[1].begin(), info[1].end());
    for (const auto& is : info[0]) {
      auto it0 = owner_group[0].find(is);
      if (it0 != owner_group[0].end() && it0->second != g) {
        SpielFatalError(absl::StrCat(
            "P0 infoset belongs to multiple public groups: ", is,
            " groups=", it0->second, " and ", g));
      }
      owner_group[0][is] = g;
    }
    for (const auto& is : info[1]) {
      auto it1 = owner_group[1].find(is);
      if (it1 != owner_group[1].end() && it1->second != g) {
        SpielFatalError(absl::StrCat(
            "P1 infoset belongs to multiple public groups: ", is,
            " groups=", it1->second, " and ", g));
      }
      owner_group[1][is] = g;
    }
  }

  // Solve FG once per (player, target-group): 2 * 30 solves.
  std::unordered_map<std::string, TabularPolicy> per_group_pol[2];
  for (int rp = 0; rp < 2; ++rp) {
    for (const auto& target_group : groups) {
      const std::string kTarget = "__classic_target__";
      const std::string kOther = "__classic_other__";
      std::unordered_map<std::string, std::vector<std::string>> bbg;
      for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
        const std::string bucket = (pub_obs == target_group) ? kTarget : kOther;
        for (const auto& root : roots) bbg[bucket].push_back(root->HistoryString());
      }
      auto trunk_ptr = std::make_shared<TabularPolicy>(trunk_policy);
      auto fg = CreateFullGadgetGame(
          game, trunk_ptr, rp, kTarget, bbg, FullGadgetGame::Mode::kTrunk,
          {}, {}, /*enumerate_boundary_portfolios=*/true);
      algorithms::ortools::SequenceFormLpSpecification sp(*fg, "CLP", false);
      sp.SpecifyLinearProgram(rp);
      double v = sp.Solve();
      per_group_pol[rp][target_group] = sp.OptimalPolicy(rp, true);
      std::cout << "  solved rp=" << rp << " target_group=" << target_group
                << " value=" << v << std::endl;
    }
  }

  // Stitch all group slices together exactly once.
  TabularPolicy stitched = uniform_base;
  for (const auto& is : all_infosets[0]) stitched.SetStatePolicy(is, full_ne.GetStatePolicy(is));
  for (const auto& is : all_infosets[1]) stitched.SetStatePolicy(is, full_ne.GetStatePolicy(is));
  for (const auto& is : trunk_all_infosets) stitched.SetStatePolicy(is, trunk_policy.GetStatePolicy(is));

  for (int pl = 0; pl < 2; ++pl) {
    for (const auto& g : groups) {
      auto itp = per_group_pol[pl].find(g);
      if (itp == per_group_pol[pl].end()) {
        SpielFatalError(absl::StrCat("Missing per-group policy for P", pl, " group=", g));
      }
      auto ita = allowed_by_group[pl].find(g);
      if (ita == allowed_by_group[pl].end()) continue;
      StitchPlayerFromFGPolicy(itp->second, pl, ita->second, trunk_all_infosets, &stitched);
    }
  }

  int missing_nontrunk_p0 = 0, missing_nontrunk_p1 = 0;
  for (const auto& is : all_infosets[0]) {
    if (trunk_all_infosets.count(is)) continue;
    if (owner_group[0].count(is) == 0) continue;
    if (stitched.GetStatePolicy(is).empty()) ++missing_nontrunk_p0;
  }
  for (const auto& is : all_infosets[1]) {
    if (trunk_all_infosets.count(is)) continue;
    if (owner_group[1].count(is) == 0) continue;
    if (stitched.GetStatePolicy(is).empty()) ++missing_nontrunk_p1;
  }
  std::cout << "  missing non-trunk stitched rows P0/P1 = "
            << missing_nontrunk_p0 << "/" << missing_nontrunk_p1 << std::endl;
  SPIEL_CHECK_EQ(missing_nontrunk_p0, 0);
  SPIEL_CHECK_EQ(missing_nontrunk_p1, 0);

  FillMissingFromReference(&stitched, full_ne, all_infosets[0], all_infosets[1]);
  std::cout << "  classic single-target stitched exploitability = "
            << algorithms::Exploitability(*game, stitched) << std::endl;
  std::cout << "RunClassicSingleTargetStitchRecheck DONE" << std::endl;
}

void RunP1PerSubgameLockRecheck() {
  std::cout << "RunP1PerSubgameLockRecheck..." << std::endl;
  auto game = LoadGame("leduc_poker");
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto groups = SortedGroups(decomp);
  auto [full_ne, full_val] = algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  std::cout << "  THE_NE exploitability = "
            << algorithms::Exploitability(*game, full_ne) << std::endl;

  TabularPolicy uniform_base(*game);
  auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::unordered_set<std::string> trunk_all_infosets;
  std::unordered_set<std::string> trunk_infosets_pl[2];
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& is : info_states) {
      trunk_all_infosets.insert(is);
      trunk_infosets_pl[player].insert(is);
    }
  }

  auto all_infosets = CollectAllInfoStates(*game);
  std::unordered_map<std::string, std::unordered_set<std::string>> allowed_by_group_p1;
  for (const auto& g : groups) {
    auto it = decomp.grouped_subgames.find(g);
    if (it == decomp.grouped_subgames.end()) continue;
    auto info = CollectSubgameInfoStatesPerPlayer(it->second);
    allowed_by_group_p1[g].insert(info[1].begin(), info[1].end());
  }

  algorithms::ortools::SequenceFormLpSpecification full_spec(*game, "CLP", false);
  const double strat_eps = algorithms::ortools::kStrategyEpsilon;

  auto merge_players = [&](const TabularPolicy& p0, const TabularPolicy& p1) {
    TabularPolicy out = uniform_base;
    for (const auto& is : all_infosets[0]) out.SetStatePolicy(is, p0.GetStatePolicy(is));
    for (const auto& is : all_infosets[1]) out.SetStatePolicy(is, p1.GetStatePolicy(is));
    return out;
  };

  auto policy_max_l1 = [&](const TabularPolicy& a, const TabularPolicy& b, int pl) {
    double m = 0.0;
    for (const auto& is : all_infosets[pl]) {
      auto pa = a.GetStatePolicy(is);
      auto pb = b.GetStatePolicy(is);
      if (pa.empty() || pb.empty()) continue;
      double s = 0.0;
      for (size_t i = 0; i < pa.size(); ++i) {
        s += std::abs(pa[i].second - pb[i].second);
      }
      m = std::max(m, s);
    }
    return m;
  };

  // Constrained re-solve on full game: fix P0 trunk (NE), fix P1 trunk (NE) + P1
  // target slice (FG extract); alternate LP solves until near convergence.
  auto constrained_resolve = [&](const std::unordered_set<std::string>& p1_target_infos,
                                 const TabularPolicy& p1_target_probs_source) {
    TabularPolicy p0_cur = full_ne;
    TabularPolicy p1_cur = full_ne;
    for (const auto& is : p1_target_infos) {
      p1_cur.SetStatePolicy(is, p1_target_probs_source.GetStatePolicy(is));
    }
    const int kMaxIters = 25;
    const double kTol = 1e-9;
    for (int it = 0; it < kMaxIters; ++it) {
      TabularPolicy fix_p0;
      for (const auto& is : trunk_infosets_pl[0]) {
        fix_p0.SetStatePolicy(is, full_ne.GetStatePolicy(is));
      }
      full_spec.SpecifyLinearProgram(0);
      algorithms::ortools::RecursivelyRefineSpecFixStrategyWithPolicy(
          full_spec.trees()[0]->mutable_root(), fix_p0, &full_spec, strat_eps);
      full_spec.Solve();
      TabularPolicy p0_next = full_spec.OptimalPolicy(0, true);

      TabularPolicy fix_p1;
      for (const auto& is : trunk_infosets_pl[1]) {
        fix_p1.SetStatePolicy(is, full_ne.GetStatePolicy(is));
      }
      for (const auto& is : p1_target_infos) {
        fix_p1.SetStatePolicy(is, p1_target_probs_source.GetStatePolicy(is));
      }
      full_spec.SpecifyLinearProgram(1);
      algorithms::ortools::RecursivelyRefineSpecFixStrategyWithPolicy(
          full_spec.trees()[1]->mutable_root(), fix_p1, &full_spec, strat_eps);
      full_spec.Solve();
      TabularPolicy p1_next = full_spec.OptimalPolicy(1, true);

      double d0 = policy_max_l1(p0_cur, p0_next, 0);
      double d1 = policy_max_l1(p1_cur, p1_next, 1);
      p0_cur = std::move(p0_next);
      p1_cur = std::move(p1_next);
      if (d0 < kTol && d1 < kTol) break;
    }
    TabularPolicy out = merge_players(p0_cur, p1_cur);
    FillMissingFromReference(&out, full_ne, all_infosets[0], all_infosets[1]);
    NormalizeNonnegativePolicy(&out, all_infosets);
    return out;
  };

  std::vector<std::pair<std::string, double>> per_group_exp;
  std::vector<std::pair<std::string, double>> per_group_exp_resolved;
  for (const auto& target_group : groups) {
    const std::string kTarget = "__p1_target__";
    const std::string kOther = "__p1_other__";
    std::unordered_map<std::string, std::vector<std::string>> bbg;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      const std::string bucket = (pub_obs == target_group) ? kTarget : kOther;
      for (const auto& root : roots) bbg[bucket].push_back(root->HistoryString());
    }

    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    auto fg = CreateFullGadgetGame(
        game, trunk_ptr, /*resolving_player=*/1, kTarget, bbg,
        FullGadgetGame::Mode::kTrunk,
        {}, {}, /*enumerate_boundary_portfolios=*/true);

    algorithms::ortools::SequenceFormLpSpecification sp(*fg, "CLP", false);
    sp.SpecifyLinearProgram(1);
    double v1 = sp.Solve();
    TabularPolicy pol1 = sp.OptimalPolicy(1, true);

    // (d) Naive stitch + exploitability (no re-solve).
    TabularPolicy locked_profile = full_ne;
    StitchPlayerFromFGPolicy(pol1, 1, allowed_by_group_p1[target_group],
                             trunk_all_infosets, &locked_profile);
    FillMissingFromReference(&locked_profile, full_ne, all_infosets[0], all_infosets[1]);
    NormalizeNonnegativePolicy(&locked_profile, all_infosets);
    double exp = algorithms::Exploitability(*game, locked_profile);

    // (d') Full-game constrained LP: fix P0 trunk, P1 trunk + target slice; iterate.
    TabularPolicy resolved =
        constrained_resolve(allowed_by_group_p1[target_group], locked_profile);
    double exp_resolved = algorithms::Exploitability(*game, resolved);

    per_group_exp.push_back({target_group, exp});
    per_group_exp_resolved.push_back({target_group, exp_resolved});

    std::cout << "  target_group=" << target_group
              << " fg_p1_value=" << v1
              << " stitched_only_exp=" << exp
              << " after_constrained_lp_exp=" << exp_resolved << std::endl;
  }

  auto summarize = [](const std::vector<std::pair<std::string, double>>& rows,
                      const char* label) {
    std::vector<std::pair<std::string, double>> sorted = rows;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    double mean = 0.0;
    for (const auto& [g, e] : sorted) mean += e;
    mean /= std::max(1, static_cast<int>(sorted.size()));
    std::cout << "  Summary " << label << ": n=" << sorted.size()
              << " mean=" << mean
              << " max=" << (sorted.empty() ? 0.0 : sorted.front().second)
              << " min=" << (sorted.empty() ? 0.0 : sorted.back().second)
              << std::endl;
  };

  summarize(per_group_exp, "P1-per-target stitched-only");
  summarize(per_group_exp_resolved, "P1-per-target after constrained LP");

  std::sort(per_group_exp_resolved.begin(), per_group_exp_resolved.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::cout << "  Top-5 worst after constrained LP:" << std::endl;
  for (int i = 0; i < std::min(5, static_cast<int>(per_group_exp_resolved.size())); ++i) {
    std::cout << "    [" << (i + 1) << "] exp=" << per_group_exp_resolved[i].second
              << " group=" << per_group_exp_resolved[i].first << std::endl;
  }
  std::cout << "RunP1PerSubgameLockRecheck DONE" << std::endl;
}

// =============================================================================
// Focused diagnostic: isolate the stitching failure
// =============================================================================
void RunStitchingDiagnostic() {
  std::cout << "=== RunStitchingDiagnostic ===" << std::endl;
  auto game = LoadGame("leduc_poker");
  TabularPolicy uniform_base(*game);
  auto decomp = DecomposeGameStructureAtRound(game, 3);
  auto groups = SortedGroups(decomp);
  auto [full_ne, full_val] =
      algorithms::ortools::MakeEquilibriumPolicy(*game, true);
  double full_ne_exp = algorithms::Exploitability(*game, full_ne);
  std::cout << "  THE_NE exp = " << full_ne_exp << std::endl;
  std::cout << "  groups = " << groups.size() << std::endl;

  auto trunk_infostates = CollectInfoStateStringsBeforeRound(*game, 3);
  std::unordered_set<std::string> trunk_all_infosets;
  for (const auto& [player, info_states] : trunk_infostates) {
    for (const auto& is : info_states) trunk_all_infosets.insert(is);
  }
  auto all_infosets = CollectAllInfoStates(*game);

  // Build per-group allowed infosets and verify no overlap
  std::unordered_map<std::string, std::unordered_set<std::string>>
      allowed_by_group[2];
  for (const auto& g : groups) {
    auto it = decomp.grouped_subgames.find(g);
    if (it == decomp.grouped_subgames.end()) continue;
    auto info = CollectSubgameInfoStatesPerPlayer(it->second);
    allowed_by_group[0][g] = info[0];
    allowed_by_group[1][g] = info[1];
  }

  // Check for overlap between groups
  std::cout << "\n--- [D1] Check info state overlap across groups ---"
            << std::endl;
  for (int pl = 0; pl < 2; ++pl) {
    std::unordered_map<std::string, std::string> owner;
    int overlaps = 0;
    for (const auto& g : groups) {
      for (const auto& is : allowed_by_group[pl][g]) {
        auto it = owner.find(is);
        if (it != owner.end() && it->second != g) {
          if (overlaps < 5)
            std::cout << "  OVERLAP P" << pl << " is=" << is
                      << " groups=" << it->second << " and " << g << std::endl;
          ++overlaps;
        }
        owner[is] = g;
      }
    }
    std::cout << "  P" << pl << " overlapping infosets: " << overlaps
              << std::endl;
  }

  // -----------------------------------------------------------------------
  // D2: All-targets baseline: FG game with NO MVS boundaries.
  //     Every boundary state is in the target group. This is equivalent to
  //     the original game with locked resolving player.
  //     Stitch result must be ~0 exploitability.
  // -----------------------------------------------------------------------
  std::cout << "\n--- [D2] All-targets baseline (no MVS) ---" << std::endl;
  {
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    const std::string kAll = "__all__";
    std::unordered_map<std::string, std::vector<std::string>> bbg;
    for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
      for (const auto& root : roots)
        bbg[kAll].push_back(root->HistoryString());
    }

    std::unordered_set<std::string> all_subgame[2];
    for (const auto& g : groups) {
      all_subgame[0].insert(allowed_by_group[0][g].begin(),
                            allowed_by_group[0][g].end());
      all_subgame[1].insert(allowed_by_group[1][g].begin(),
                            allowed_by_group[1][g].end());
    }

    for (int rp = 0; rp < 2; ++rp) {
      auto fg = CreateFullGadgetGame(game, trunk_ptr, rp, kAll, bbg,
                                     FullGadgetGame::Mode::kTrunk, {}, {},
                                     false);
      algorithms::ortools::SequenceFormLpSpecification sp(*fg, "CLP", false);
      sp.SpecifyLinearProgram(rp);
      double v = sp.Solve();
      TabularPolicy pol = sp.OptimalPolicy(rp, true);

      TabularPolicy stitched = full_ne;
      StitchPlayerFromFGPolicy(pol, rp, all_subgame[rp], trunk_all_infosets,
                               &stitched);
      FillMissingFromReference(&stitched, full_ne, all_infosets[0],
                               all_infosets[1]);
      double exp = algorithms::Exploitability(*game, stitched);
      std::cout << "  rp=" << rp << " value=" << v << " stitched_exp=" << exp
                << std::endl;

      // Compare strategy vs NE at each subgame info state
      int diff_count = 0;
      double max_l1 = 0.0;
      for (const auto& is : all_subgame[rp]) {
        auto ne_ap = full_ne.GetStatePolicy(is);
        auto st_ap = stitched.GetStatePolicy(is);
        double l1 = L1PolicyDelta(ne_ap, st_ap);
        if (l1 > 1e-10) {
          ++diff_count;
          max_l1 = std::max(max_l1, l1);
        }
      }
      std::cout << "    diff_from_NE: count=" << diff_count
                << " max_l1=" << max_l1 << std::endl;
    }
  }

  // -----------------------------------------------------------------------
  // D3: Per-group single replacement.
  //     For each (player, group), solve FG, replace ONLY that group/player
  //     in the NE, measure exploitability.
  // -----------------------------------------------------------------------
  std::cout << "\n--- [D3] Per-group single replacement ---" << std::endl;
  {
    auto trunk_ptr = std::make_shared<TabularPolicy>(full_ne);
    double max_single_exp = 0.0;
    int num_above_1e6 = 0;

    // Store all per-group policies for later use in D4/D5
    std::unordered_map<std::string, TabularPolicy> per_group_pol[2];

    for (int rp = 0; rp < 2; ++rp) {
      for (const auto& target_group : groups) {
        const std::string kT = "__d3_target__";
        const std::string kO = "__d3_other__";
        std::unordered_map<std::string, std::vector<std::string>> bbg;
        for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
          const std::string bucket =
              (pub_obs == target_group) ? kT : kO;
          for (const auto& root : roots)
            bbg[bucket].push_back(root->HistoryString());
        }

        auto fg = CreateFullGadgetGame(game, trunk_ptr, rp, kT, bbg,
                                       FullGadgetGame::Mode::kTrunk, {}, {},
                                       true);
        algorithms::ortools::SequenceFormLpSpecification sp(*fg, "CLP", false);
        sp.SpecifyLinearProgram(rp);
        double v = sp.Solve();
        TabularPolicy pol = sp.OptimalPolicy(rp, true);
        per_group_pol[rp][target_group] = pol;

        // Stitch ONLY this group for this player into NE
        TabularPolicy stitched = full_ne;
        StitchPlayerFromFGPolicy(pol, rp, allowed_by_group[rp][target_group],
                                 trunk_all_infosets, &stitched);
        FillMissingFromReference(&stitched, full_ne, all_infosets[0],
                                 all_infosets[1]);
        double exp = algorithms::Exploitability(*game, stitched);
        if (exp > 1e-6) {
          ++num_above_1e6;
          if (num_above_1e6 <= 10)
            std::cout << "  rp=" << rp << " group=" << target_group
                      << " val=" << v << " single_exp=" << exp << std::endl;
        }
        max_single_exp = std::max(max_single_exp, exp);
      }
    }
    std::cout << "  max single-group exp = " << max_single_exp
              << " (above 1e-6: " << num_above_1e6 << "/" << (groups.size() * 2)
              << ")" << std::endl;

    // -----------------------------------------------------------------------
    // D4: Full stitching (all groups, both players) - reproduce the 0.004 bug
    // -----------------------------------------------------------------------
    std::cout << "\n--- [D4] Full stitching (all groups, both players) ---"
              << std::endl;
    {
      TabularPolicy stitched = full_ne;
      for (int pl = 0; pl < 2; ++pl) {
        for (const auto& g : groups) {
          StitchPlayerFromFGPolicy(per_group_pol[pl][g], pl,
                                   allowed_by_group[pl][g],
                                   trunk_all_infosets, &stitched);
        }
      }
      FillMissingFromReference(&stitched, full_ne, all_infosets[0],
                               all_infosets[1]);
      double exp_full = algorithms::Exploitability(*game, stitched);
      std::cout << "  full stitch exp = " << exp_full << std::endl;

      // How many info states differ from NE?
      for (int pl = 0; pl < 2; ++pl) {
        int diff_count = 0;
        double sum_l1 = 0.0, max_l1 = 0.0;
        std::string worst_is;
        for (const auto& is : all_infosets[pl]) {
          if (trunk_all_infosets.count(is)) continue;
          auto ne_ap = full_ne.GetStatePolicy(is);
          auto st_ap = stitched.GetStatePolicy(is);
          double l1 = L1PolicyDelta(ne_ap, st_ap);
          if (l1 > 1e-10) {
            ++diff_count;
            sum_l1 += l1;
            if (l1 > max_l1) {
              max_l1 = l1;
              worst_is = is;
            }
          }
        }
        std::cout << "  P" << pl << " diff: count=" << diff_count
                  << " sum_l1=" << sum_l1 << " max_l1=" << max_l1
                  << " worst=" << worst_is << std::endl;
      }
    }

    // -----------------------------------------------------------------------
    // D5: Incremental stitching - add groups one at a time for P0
    //     to find where exploitability starts growing.
    // -----------------------------------------------------------------------
    std::cout << "\n--- [D5] Incremental stitching (P0 only) ---" << std::endl;
    {
      TabularPolicy stitched = full_ne;
      for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
        const auto& g = groups[gi];
        StitchPlayerFromFGPolicy(per_group_pol[0][g], 0,
                                 allowed_by_group[0][g], trunk_all_infosets,
                                 &stitched);
        TabularPolicy temp = stitched;
        FillMissingFromReference(&temp, full_ne, all_infosets[0],
                                 all_infosets[1]);
        double exp = algorithms::Exploitability(*game, temp);
        if (gi < 5 || exp > 1e-6)
          std::cout << "  after P0 group[" << gi << "] (" << g
                    << ") exp=" << exp << std::endl;
      }
    }

    // -----------------------------------------------------------------------
    // D6: Incremental stitching - add groups one at a time for P1
    // -----------------------------------------------------------------------
    std::cout << "\n--- [D6] Incremental stitching (P1 only) ---" << std::endl;
    {
      TabularPolicy stitched = full_ne;
      for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
        const auto& g = groups[gi];
        StitchPlayerFromFGPolicy(per_group_pol[1][g], 1,
                                 allowed_by_group[1][g], trunk_all_infosets,
                                 &stitched);
        TabularPolicy temp = stitched;
        FillMissingFromReference(&temp, full_ne, all_infosets[0],
                                 all_infosets[1]);
        double exp = algorithms::Exploitability(*game, temp);
        if (gi < 5 || exp > 1e-6)
          std::cout << "  after P1 group[" << gi << "] (" << g
                    << ") exp=" << exp << std::endl;
      }
    }

    // -----------------------------------------------------------------------
    // D7: Compare FG-solved strategy vs NE per info state, by group
    //     Shows which groups produce strategies that differ from NE
    // -----------------------------------------------------------------------
    std::cout << "\n--- [D7] Per-group strategy diff from NE ---" << std::endl;
    for (int rp = 0; rp < 2; ++rp) {
      for (const auto& g : groups) {
        const auto& pol = per_group_pol[rp][g];
        int diff_count = 0;
        double max_l1 = 0.0;
        const std::string prefix = "full_F:subgame:";
        const std::string expected_tag =
            absl::StrCat("full_F:subgame:P", rp, ":");
        for (const auto& [sub_is, ap] : pol.PolicyTable()) {
          if (sub_is.compare(0, prefix.size(), prefix) != 0) continue;
          if (sub_is.compare(0, expected_tag.size(), expected_tag) != 0)
            continue;
          std::string orig = sub_is.substr(prefix.size());
          if (orig.size() > 3 && orig[0] == 'P' &&
              (orig[1] == '0' || orig[1] == '1') && orig[2] == ':')
            orig = orig.substr(3);
          if (allowed_by_group[rp][g].count(orig) == 0) continue;
          auto ne_ap = full_ne.GetStatePolicy(orig);
          if (ne_ap.empty()) continue;
          // Clamp and normalize the FG policy for fair comparison
          ActionsAndProbs clamped;
          double sum = 0.0;
          for (const auto& [a, p] : ap) {
            double cp = std::max(0.0, p);
            clamped.push_back({a, cp});
            sum += cp;
          }
          if (sum > 0.0)
            for (auto& [a, p] : clamped) p /= sum;
          double l1 = L1PolicyDelta(ne_ap, clamped);
          if (l1 > 1e-10) {
            ++diff_count;
            max_l1 = std::max(max_l1, l1);
          }
        }
        if (diff_count > 0 || max_l1 > 1e-6)
          std::cout << "  rp=" << rp << " group=" << g
                    << " diff_count=" << diff_count << " max_l1=" << max_l1
                    << std::endl;
      }
    }

    // -----------------------------------------------------------------------
    // D8: Check if stitched strategy is still an NE in the FG game.
    //     For each group, build the FG game with that target, check if the
    //     stitched strategy (restricted to that FG game) is a NE.
    // -----------------------------------------------------------------------
    std::cout << "\n--- [D8] NE check: stitched strategy in each FG game ---"
              << std::endl;
    {
      // Build the full stitched policy
      TabularPolicy stitched = full_ne;
      for (int pl = 0; pl < 2; ++pl) {
        for (const auto& g : groups) {
          StitchPlayerFromFGPolicy(per_group_pol[pl][g], pl,
                                   allowed_by_group[pl][g],
                                   trunk_all_infosets, &stitched);
        }
      }
      FillMissingFromReference(&stitched, full_ne, all_infosets[0],
                               all_infosets[1]);

      // For each group, create the FG game and evaluate the stitched strategy
      // within it (by computing exploitability of the combined FG policy).
      int tested = 0;
      for (int rp = 0; rp < 2; ++rp) {
        for (const auto& target_group : groups) {
          if (tested >= 6) break;  // Only check a few
          const std::string kT = "__d8_t__";
          const std::string kO = "__d8_o__";
          std::unordered_map<std::string, std::vector<std::string>> bbg;
          for (const auto& [pub_obs, roots] : decomp.grouped_subgames) {
            const std::string bucket =
                (pub_obs == target_group) ? kT : kO;
            for (const auto& root : roots)
              bbg[bucket].push_back(root->HistoryString());
          }

          auto fg = CreateFullGadgetGame(game, trunk_ptr, rp, kT, bbg,
                                         FullGadgetGame::Mode::kTrunk, {}, {},
                                         true);
          // The stitched policy in the FG game: map original infosets to FG
          // infosets. Since the FG game extracts strategies back via prefix,
          // we can evaluate by computing the FG LP value for BOTH players.
          algorithms::ortools::SequenceFormLpSpecification sp(*fg, "CLP",
                                                              false);
          sp.SpecifyLinearProgram(rp);
          double v_rp = sp.Solve();
          sp.SpecifyLinearProgram(1 - rp);
          double v_opp = sp.Solve();
          double fg_exp = (v_rp + v_opp) / 2.0;
          // In zero-sum: exploitability = (v_rp + v_opp) / 2 should be ~0
          // if both LP solutions are consistent
          std::cout << "  rp=" << rp << " group=" << target_group
                    << " v_rp=" << v_rp << " v_opp=" << v_opp
                    << " fg_gap=" << (v_rp + v_opp) << std::endl;
          ++tested;
        }
      }
    }
  }

  std::cout << "\n=== RunStitchingDiagnostic DONE ===" << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::string mode = "diag";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--test_r_only") mode = "test_r";
    else if (std::string(argv[i]) == "--classic_single_target_only") mode = "classic";
    else if (std::string(argv[i]) == "--p1_per_subgame_lock_only") mode = "p1lock";
    else if (std::string(argv[i]) == "--diag") mode = "diag";
    else if (std::string(argv[i]) == "--half_split") mode = "half_split";
  }
  if (mode == "p1lock") {
    open_spiel::RunP1PerSubgameLockRecheck();
  } else if (mode == "classic") {
    open_spiel::RunClassicSingleTargetStitchRecheck();
  } else if (mode == "test_r") {
    open_spiel::RunStitchingOnlyTestRStrict();
  } else if (mode == "diag") {
    open_spiel::RunStitchingDiagnostic();
  } else {
    open_spiel::RunLeducHalfSplitExperiment();
  }
  return 0;
}

