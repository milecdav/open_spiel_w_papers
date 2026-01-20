macro(add_paper_executable NAME SOURCES)
    add_executable(${NAME} ${SOURCES} $<TARGET_OBJECTS:open_spiel_ortools>
            ${OPEN_SPIEL_OBJECTS})
    target_link_libraries(${NAME})
endmacro()

add_paper_executable(dlr_main dlr_main.cc)
