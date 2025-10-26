set(NAME omp)

file(GLOB SRC_FILES "*.cpp")

add_executable(${NAME} 
    ${SRC_FILES} 
    ${COMMON_PATH}/logger.cpp
    ${COMMON_PATH}/common.cpp
    ${COMMON_PATH}/task.cpp
)

find_package(OpenMP REQUIRED)
target_link_libraries(${NAME} PUBLIC OpenMP::OpenMP_CXX)

include_directories(${NAME} ${COMMON_PATH})