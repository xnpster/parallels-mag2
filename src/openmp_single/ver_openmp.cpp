#include <iostream>
#include <cstdlib>
#include <omp.h>

#include "logger.h"
#include "common.h"
#include "task.h"

using namespace std;

static double *computed_data[2];

static size_t x_sz, y_sz, z_sz;
static size_t x_stride, y_stride, z_stride;

void SetupData()
{
    x_sz = s_task_params.N + 1 + 2;
    y_sz = x_sz;
    z_sz = x_sz;

    z_stride = 1;
    y_stride = z_stride * z_sz;
    x_stride = y_stride * y_sz;

    for (int t = 0; t < sizeof(computed_data) / sizeof(*computed_data); t++)
        computed_data[t] = (double *)calloc(x_sz * y_sz * z_sz, sizeof(double));

    LogWrite("Total elements: " + to_string(x_sz * y_sz * z_sz) + "(" +
             to_string(x_sz * y_sz * z_sz * sizeof(double) / double(8 * 1024 * 1024)) + " MB)");
}

void FreeData()
{
    for (int t = 0; t < sizeof(computed_data) / sizeof(*computed_data); t++)
    {
        free(computed_data[t]);
        computed_data[t] = nullptr;
    }

    LogWrite("Memory freed");
}

int main(int argc, char **argv)
{
    LogInit(argc, argv);
    ReadOmpOptions(argc, argv);
    TaskParamsInit(argc, argv);
    SetupData();

    size_t t = 0;
    double ts = 0;

    LogWrite("Fill u0, u1...");
    
    for (; t < 2; t++)
    {
        LogWrite("    t = " + to_string(t) + ", ts = " + to_string(ts));
        FillAnalytical(computed_data[t % 2], ts, 0, 0, 0, x_sz, y_sz, z_sz, x_stride, y_stride);
        ts += s_task_params.tau;
    }

    LogWrite("Fill done, iterating...");
    while (t < s_task_params.K)
    {
        t++;
        ts += s_task_params.tau;
        LogWrite("    t = " + to_string(t) + ", ts = " + to_string(ts));

        auto *current = computed_data[t % 2];
        auto *prev = computed_data[(t - 1) % 2];

        #pragma omp parallel for collapse(3)
        for (size_t i = 1; i <= s_task_params.N + 1; i++)
            for (size_t j = 1; j <= s_task_params.N + 1; j++)
                for (size_t k = 1; k <= s_task_params.N + 1; k++)
                    MakeStep(current, prev, i, j, k, x_stride, y_stride);

        #pragma omp parallel for collapse(2)
        for (size_t j = 1; j <= s_task_params.N + 1; j++)
        {
            for (size_t k = 1; k <= s_task_params.N + 1; k++)
            {
                current[0 * x_stride + j * y_stride + k] = current[s_task_params.N * x_stride + j * y_stride + k];
                current[(s_task_params.N + 2) * x_stride + j * y_stride + k] = current[2 * x_stride + j * y_stride + k];
            }
        }

        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i <= s_task_params.N + 1; i++)
        {
            for (size_t k = 1; k <= s_task_params.N + 1; k++)
            {
                current[i * x_stride + 0 * y_stride + k] = current[i * x_stride + s_task_params.N * y_stride + k];
                current[i * x_stride + (s_task_params.N + 2) * y_stride + k] = current[i * x_stride + 2 * y_stride + k];
            }
        }

        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i <= s_task_params.N + 1; i++)
        {
            for (size_t j = 1; j <= s_task_params.N + 1; j++)
            {
                current[i * x_stride + j * y_stride + 1] = 0;
                current[i * x_stride + j * y_stride + s_task_params.N + 1] = 0;
            }
        }
    }

    LogWrite("Compute error...");

    FillAnalytical(computed_data[(t + 1) % 2], ts, 0, 0, 0, x_sz, y_sz, z_sz, x_stride, y_stride);
    auto ae = MaxAbsoluteError(computed_data[t % 2], computed_data[(t + 1) % 2], x_sz, y_sz, z_sz, x_stride, y_stride);
    LogWrite("    max absolute error: " + to_string(ae));

    LogWrite("Free data...");
    FreeData();
}