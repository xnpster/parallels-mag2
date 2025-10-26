#include "task.h"
#include "logger.h"

#include <string>
#include <unordered_map>
#include <math.h>

using namespace std;

TaskParams s_task_params;
SolutionConsts s_solution_const;

void TaskParamsInit(int argc, char **argv)
{
    s_task_params.K = 10;
    s_task_params.N = 128;

    s_task_params.T = 1;

    for (int i = 0; i < TaskParams::SPACE_DIM; i++)
        s_task_params.L[i] = 1;

    static const unordered_map<string, double *> float_options = {
        {"-T", &s_task_params.T},
        {"-Lx", &s_task_params.L[0]},
        {"-Ly", &s_task_params.L[1]},
        {"-Lz", &s_task_params.L[2]},
    };

    static const unordered_map<string, unsigned long long*> int_options = {
        {"-K", &s_task_params.K},
        {"-N", &s_task_params.N}
    };

    for (int i = 1; i < argc; i++)
    {
        auto option = string(argv[i]);
        
        auto it1 = float_options.find(option);
        if (it1 != float_options.end())
        {
            *(it1->second) = strtod(argv[i + 1], nullptr);
            i++;
            continue;
        }

        auto it2 = int_options.find(option);
        if (it2 != int_options.end())
        {
            *(it2->second) = strtoull(argv[i + 1], nullptr, 10);
            i++;
            continue;
        }
    }

    s_task_params.tau = s_task_params.T / s_task_params.K;

    for (int i = 0; i < TaskParams::SPACE_DIM; i++)
    {
        s_task_params.step[i] = s_task_params.L[i] / s_task_params.N;
        s_task_params.step_squared[i] = s_task_params.step[i] * s_task_params.step[i];
    }

    s_task_params.a_squared = 1 / (
        4 / (s_task_params.L[0] * s_task_params.L[0]) + 
        4 / (s_task_params.L[1] * s_task_params.L[1]) + 
        1 / (s_task_params.L[2] * s_task_params.L[2])
    );

    s_task_params.c1 = s_task_params.tau * s_task_params.tau * s_task_params.a_squared;

    s_task_params.c2 = 2 * (1 - s_task_params.c1 * (
        1/s_task_params.step_squared[0] +
        1/s_task_params.step_squared[1] +
        1/s_task_params.step_squared[2]
    ));
    
    LogWrite("K: " + to_string(s_task_params.K));
    LogWrite("N: " + to_string(s_task_params.N));
    LogWrite("T: " + to_string(s_task_params.T) + ", tau: " + to_string(s_task_params.tau));
    LogWrite("Lx: " + to_string(s_task_params.L[0]) + ", step: " + to_string(s_task_params.step[0]));
    LogWrite("Ly: " + to_string(s_task_params.L[1]) + ", step: " + to_string(s_task_params.step[1]));
    LogWrite("Lz: " + to_string(s_task_params.L[2]) + ", step: " + to_string(s_task_params.step[2]));
    LogWrite("a^2: " + to_string(s_task_params.a_squared));

    s_solution_const.xc = 2 * M_PI / s_task_params.L[0];
    s_solution_const.yc = 2 * M_PI / s_task_params.L[1];
    s_solution_const.zc = 1 * M_PI / s_task_params.L[2];

    LogWrite("solution: sin(x) coef: " + to_string(s_solution_const.xc));
    LogWrite("solution: sin(y) coef: " + to_string(s_solution_const.yc));
    LogWrite("solution: sin(z) coef: " + to_string(s_solution_const.zc));
}