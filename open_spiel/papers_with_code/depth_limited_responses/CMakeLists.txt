add_library(depth_limited_responses OBJECT
        secfr.cc
        secfr.h)

macro(add_paper_executable NAME SOURCES)
    add_executable(${NAME} ${SOURCES} $<TARGET_OBJECTS:open_spiel_ortools> $<TARGET_OBJECTS:depth_limited_responses>
            ${OPEN_SPIEL_OBJECTS})
    target_link_libraries(${NAME})
endmacro()

add_paper_executable(dlr_main dlr_main.cc)
add_paper_executable(exact_sweep exact_sweep_main.cc)
