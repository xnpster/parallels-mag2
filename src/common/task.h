#pragma once

#include <cmath>
#include <cstdio>

using std::size_t;

struct TaskParams
{
	static const int SPACE_DIM = 3;

	unsigned long long K, N;
	double T, tau;
	double L[SPACE_DIM];
	double step[SPACE_DIM], step_squared[SPACE_DIM];
	double a_squared;
	double c1, c2;
};

extern TaskParams s_task_params;

void TaskParamsInit(int argc, char **argv);

struct SolutionConsts
{
	double xc, yc, zc;
};

extern SolutionConsts s_solution_const;

static double AnalyticalSolution(double x, double y, double z, double t)
{
	double r = sin(s_solution_const.xc * x + 3 * M_PI) *
	           sin(s_solution_const.yc * y + 2 * M_PI) *
	           sin(s_solution_const.zc * z) *
	           cos(M_PI * t + M_PI);

	return r;
}

static void FillAnalytical(double *data,
                           double ts,
                           double x0, double y0, double z0,
                           size_t x_sz, size_t y_sz, size_t z_sz,
                           size_t x_stride, size_t y_stride)
{
#pragma omp parallel for collapse(3) firstprivate(x0, y0, z0, x_sz, y_sz, z_sz, data, x_stride, y_stride, ts) \
            shared(s_task_params) default(none)
	for (size_t i = 1; i < x_sz - 1; i++)
	{
		for (size_t j = 1; j < y_sz - 1; j++)
		{
			for (size_t k = 1; k < z_sz - 1; k++)
			{
				double x = x0 + (i - 1) * s_task_params.step[0];
				double y = y0 + (j - 1) * s_task_params.step[1];
				double z = z0 + (k - 1) * s_task_params.step[2];

				data[i * x_stride + j * y_stride + k] = AnalyticalSolution(x, y, z, ts);
			}
		}
	}
}

static void MakeStep(double *next_d, double *curr_d,
                     size_t i, size_t j, size_t k,
                     size_t x_stride, size_t y_stride)
{

	double laplace = 0;

	size_t point_coord = i * x_stride + j * y_stride + k;

	laplace += (curr_d[point_coord - x_stride] + curr_d[point_coord + x_stride]) / s_task_params.step_squared[0];
	laplace += (curr_d[point_coord - y_stride] + curr_d[point_coord + y_stride]) / s_task_params.step_squared[1];
	laplace += (curr_d[point_coord - 1] + curr_d[point_coord + 1]) / s_task_params.step_squared[2];

	next_d[point_coord] = s_task_params.c1 * laplace + s_task_params.c2 * curr_d[point_coord] - next_d[point_coord];
}

static double MaxAbsoluteError(double *data_a, double *data_b,
                               size_t x_sz, size_t y_sz, size_t z_sz,
                               size_t x_stride, size_t y_stride)
{
	double result = 0;

#pragma omp parallel for collapse(3) reduction(max : result) \
        firstprivate(x_sz, y_sz, z_sz, data_a, data_b, x_stride, y_stride) default(none)
	for (size_t i = 1; i < x_sz - 1; i++)
	{
		for (size_t j = 1; j < y_sz - 1; j++)
		{
			for (size_t k = 1; k < z_sz - 1; k++)
			{
				size_t p = i * x_stride + j * y_stride + k;
				double diff = std::abs(data_a[p] - data_b[p]);

				if (diff > result)
					result = diff;
			}
		}
	}

	return result;
}