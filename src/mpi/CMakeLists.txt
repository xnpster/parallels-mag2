set(NAME mpi)

file(GLOB SRC_FILES "*.cpp")

add_executable(${NAME} 
    ${SRC_FILES} 
    ${COMMON_PATH}/logger.cpp
    ${COMMON_PATH}/common.cpp
    ${COMMON_PATH}/task.cpp
)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(${NAME} PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)
include_directories(${NAME} ${COMMON_PATH} ${MPI_INCLUDE_PATH})