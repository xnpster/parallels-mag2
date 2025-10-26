#include "common.h"
#include "logger.h"
#include "omp.h"

#include <string>

using namespace std;

int s_omp_threads;

void ReadOmpOptions(int argc, char** argv)
{
    s_omp_threads = 0;

    for (int i = 1; i < argc; i++)
    {
        auto option = string(argv[i]);

        if (option == "-t")
        {
            s_omp_threads = std::stoi(argv[i + 1]);
            i++;
        }
    }

    if (s_omp_threads > 0)
    {
        omp_set_num_threads(s_omp_threads);
        LogWrite("omp num threads: " + std::to_string(s_omp_threads));
    }
}