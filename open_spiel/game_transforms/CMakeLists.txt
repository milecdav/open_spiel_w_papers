add_library (game_transforms OBJECT
  continual_resolving.cc
  continual_resolving.h
  coop_to_1p.cc
  coop_to_1p.h
  efg_writer.cc
  efg_writer.h
  matrix_valued_states.cc
  matrix_valued_states.h
  misere.cc
  misere.h
  normal_form_extensive_game.cc
  normal_form_extensive_game.h
  repeated_game.cc
  repeated_game.h
  max_margin_gadget.cc
  max_margin_gadget.h
  resolving_gadget.cc
  resolving_gadget.h
  subgame_utils.cc
  subgame_utils.h
  start_at.cc
  start_at.h
  turn_based_simultaneous_game.cc
  turn_based_simultaneous_game.h
  unsafe_subgame.cc
  unsafe_subgame.h
)
target_include_directories (game_transforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(turn_based_simultaneous_game_test
               turn_based_simultaneous_game_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(turn_based_simultaneous_game_test turn_based_simultaneous_game_test)

add_executable(misere_test
               misere_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(misere_test misere_test)

add_executable(coop_to_1p_test
               coop_to_1p_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(coop_to_1p_test coop_to_1p_test)

add_executable(efg_writer_test
               efg_writer_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(efg_writer_test efg_writer_test)

add_executable(normal_form_extensive_game_test
               normal_form_extensive_game_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(normal_form_extensive_game_test normal_form_extensive_game_test)

add_executable(repeated_game_test
               repeated_game_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(repeated_game_test repeated_game_test)

add_executable(start_at_test
  start_at_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(start_at_test start_at_test)

add_executable(matrix_valued_states_test
  matrix_valued_states_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(matrix_valued_states_test matrix_valued_states_test)

add_executable(resolving_gadget_test
  resolving_gadget_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(resolving_gadget_test resolving_gadget_test)

add_executable(unsafe_subgame_test
  unsafe_subgame_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(unsafe_subgame_test unsafe_subgame_test)

add_executable(max_margin_gadget_test
  max_margin_gadget_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(max_margin_gadget_test max_margin_gadget_test)

add_executable(continual_resolving_test
  continual_resolving_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(continual_resolving_test continual_resolving_test)
