# Now we can use #include "open_spiel/spiel.h" in the paper subdirectories.
include_directories(../..)

add_subdirectory(depth_limited_responses)
add_subdirectory(public_state_cfr)
add_subdirectory(2006.08740.sound_search)
add_subdirectory(2007.14358.predictive_rm)

