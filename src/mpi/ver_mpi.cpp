#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <unistd.h>
#include <cstring>

#include "logger.h"
#include "common.h"
#include "task.h"

using namespace std;

bool log_proc;
static int dim_procs[TaskParams::SPACE_DIM];
static int cart_coords[TaskParams::SPACE_DIM];
MPI_Comm cart = MPI_COMM_NULL;
int cart_rank, cart_size;
static int cart_neigh[TaskParams::SPACE_DIM][2];

static double *computed_data[2];
static double *send_buffers[TaskParams::SPACE_DIM][2];
static double *recv_buffers[TaskParams::SPACE_DIM][2];
static size_t send_buffer_sz[TaskParams::SPACE_DIM];

static size_t sz[TaskParams::SPACE_DIM];
static size_t stride[TaskParams::SPACE_DIM];
static double s[TaskParams::SPACE_DIM];

void Init(int argc, char **argv)
{
	int world_size, world_rank;

	// basic MPI initialization
	MPI_Init(nullptr, nullptr);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// setup logging and openmp
	// calls to Log* functions without LogInit do nothing
	log_proc = world_rank == 0;
	if (log_proc)
		LogInit(argc, argv);

	ReadOmpOptions(argc, argv);
	TaskParamsInit(argc, argv);

	LogWrite("P: " + to_string(world_size));

	// split linear space into 3D grid
	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
		dim_procs[i] = 0;

	MPI_Dims_create(world_size, TaskParams::SPACE_DIM, dim_procs);
	LogWrite("Processor grid: " + to_string(dim_procs[0]) + " x " +
	         to_string(dim_procs[1]) + " x " +
	         to_string(dim_procs[2]));

	// setup Cartesian topology
	int periods[TaskParams::SPACE_DIM];
	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
		periods[i] = 1;

	int reorder = true;

	MPI_Cart_create(MPI_COMM_WORLD, TaskParams::SPACE_DIM, dim_procs, periods, reorder, &cart);

	MPI_Comm_size(cart, &cart_size);
	MPI_Comm_rank(cart, &cart_rank);

	MPI_Cart_coords(cart, cart_rank, TaskParams::SPACE_DIM, cart_coords);

	// fill ranks of neighbor nodes
	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
	{
		for (int neigh = 0; neigh < 2; neigh++)
		{
			int neigh_coords[TaskParams::SPACE_DIM];
			memcpy(neigh_coords, cart_coords, sizeof(cart_coords));
			neigh_coords[i] += (neigh == 0 ? -1 : 1);

			MPI_Cart_rank(cart, neigh_coords, &(cart_neigh[i][neigh]));
		}
	}

	// setup main data
	size_t total_points = s_task_params.N + 1;
	size_t shadow_size = 1;

	size_t total_elems = 1;

	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
	{
		size_t dim_elems = total_points / dim_procs[i];
		if (cart_coords[i] + 1 >= dim_procs[i])
			dim_elems += total_points % dim_procs[i];

		dim_elems += 2 * shadow_size;
		sz[i] = dim_elems;
		total_elems *= sz[i];
	}

	for (int t = 0; t < sizeof(computed_data) / sizeof(*computed_data); t++)
		computed_data[t] = (double *)calloc(total_elems, sizeof(double));

	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
	{
		size_t alloc_elems = total_elems / sz[i];
		for (int dir = 0; dir < 2; dir++)
		{
			if (i == 2 && ((cart_coords[i] == 0 && dir == 0) || (cart_coords[i] == dim_procs[i] - 1 && dir == 1)))
			{
				send_buffers[i][dir] = nullptr;
				recv_buffers[i][dir] = nullptr;
			}
			else
			{
				send_buffers[i][dir] = (double *)calloc(alloc_elems, sizeof(double));
				recv_buffers[i][dir] = (double *)calloc(alloc_elems, sizeof(double));
			}
		}

		send_buffer_sz[i] = alloc_elems;
	}

	LogWrite("Total elements per node: " + to_string(total_elems) + " (" +
	         to_string(total_elems * sizeof(double) / double(8 * 1024 * 1024)) + " MB) (x" +
	         to_string(sizeof(computed_data) / sizeof(*computed_data)) + ")");

	size_t acc = 1;

	// compute strides
	for (int i = TaskParams::SPACE_DIM - 1; i >= 0; i--)
	{
		stride[i] = acc;
		acc *= sz[i];
	}

	// compute coordinates of current block
	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
		s[i] = 0 + cart_coords[i] * (total_points / dim_procs[i]) * s_task_params.step[i];

	cout << "[Cart] rank: " << cart_rank << "/" << cart_size;

	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
		cout << " " << cart_coords[i];

	cout << " neighbors:";

	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
		cout << " (" << cart_neigh[i][0] << ", " << cart_neigh[i][1] << ")";

	cout << " block start: (";
	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
		cout << s[i] << " ";

	cout << ")";

	cout << " " << sz[0] << " " << sz[1] << " " << sz[2];

	cout << endl;
}

void Finalize()
{
	LogWrite("Free data...");

	for (int t = 0; t < sizeof(computed_data) / sizeof(*computed_data); t++)
	{
		free(computed_data[t]);
		computed_data[t] = nullptr;
	}

	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
	{
		for (int dir = 0; dir < 2; dir++)
		{
			free(send_buffers[i][dir]);
			send_buffers[i][dir] = nullptr;

			free(recv_buffers[i][dir]);
			recv_buffers[i][dir] = nullptr;
		}
	}

	LogWrite("Memory freed");

	LogFlush();
	LogFinalize();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

static inline void CopyEdgeToShadowBuf(double *data, double *shadow, size_t copy_idx,
                                       size_t sz1, size_t sz2, size_t sz_copy,
                                       size_t stride1, size_t stride2, size_t stride_copy)
{
	size_t shadow_stride1 = stride1 >= stride_copy ? stride1 / sz_copy : stride1;
	size_t shadow_stride2 = stride2 >= stride_copy ? stride2 / sz_copy : stride2;

#pragma omp parallel for collapse(2) default(none) \
        firstprivate(shadow_stride1, shadow_stride2, data, shadow, sz1, sz2, sz_copy, stride1, stride2, stride_copy, copy_idx)
	for (size_t a = 0; a < sz1; a++)
		for (size_t b = 0; b < sz2; b++)
			shadow[a * shadow_stride1 + b * shadow_stride2] = data[a * stride1 + b * stride2 + copy_idx * stride_copy];
}

static inline void CopyEdgeFromShadowBuf(double *data, double *shadow, size_t copy_idx,
                                         size_t sz1, size_t sz2, size_t sz_copy,
                                         size_t stride1, size_t stride2, size_t stride_copy)
{
	size_t shadow_stride1 = stride1 >= stride_copy ? stride1 / sz_copy : stride1;
	size_t shadow_stride2 = stride2 >= stride_copy ? stride2 / sz_copy : stride2;

#pragma omp parallel for collapse(2) default(none) \
        firstprivate(shadow_stride1, shadow_stride2, data, shadow, copy_idx, sz1, sz2, sz_copy, stride1, stride2, stride_copy)
	for (size_t a = 0; a < sz1; a++)
		for (size_t b = 0; b < sz2; b++)
			data[a * stride1 + b * stride2 + copy_idx * stride_copy] = shadow[a * shadow_stride1 + b * shadow_stride2];
}

void UpdateShadow(double *data)
{
	LogWrite("   updating shadow...");
	// copy data to send buffers
	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
	{
		for (int neigh = 0; neigh < 2; neigh++)
		{
			double *send_buffer = send_buffers[i][neigh];

			if (!send_buffer)
				continue;

			size_t dim_a = (i + 1) % 3;
			size_t dim_b = (i + 2) % 3;

			if (dim_b < dim_a)
				swap(dim_a, dim_b);

			int copy_idx = neigh == 0 ? 1 : sz[i] - 2;

			if ((cart_coords[i] == 0 || cart_coords[i] == dim_procs[i] - 1) && (i == 0 || i == 1))
			{
				// periodic edges
				if (cart_coords[i] == 0 && neigh == 0)
					copy_idx = 2;
				else if (cart_coords[i] == dim_procs[i] - 1 && neigh > 0)
					copy_idx = sz[i] - 3;
			}

			CopyEdgeToShadowBuf(data, send_buffer, copy_idx,
			                    sz[dim_a], sz[dim_b], sz[i],
			                    stride[dim_a], stride[dim_b], stride[i]);
		}
	}

	MPI_Request requests[TaskParams::SPACE_DIM * 2 * 2];
	int n_requests = 0, completed = 0;

	MPI_Barrier(cart);

	for (int i = 0; i < TaskParams::SPACE_DIM; i++)
	{
		for (int neigh = 0; neigh < 2; neigh++)
		{
			double *send_buffer = send_buffers[i][neigh];
			double *recv_buffer = recv_buffers[i][neigh];

			if (send_buffer && recv_buffer)
			{
				MPI_Isend(send_buffer, send_buffer_sz[i], MPI_DOUBLE, cart_neigh[i][neigh],
				          neigh, cart, &requests[n_requests]);

				n_requests++;

				MPI_Irecv(recv_buffer, send_buffer_sz[i], MPI_DOUBLE, cart_neigh[i][neigh],
				          1 - neigh, cart, &requests[n_requests]);

				n_requests++;
			}
			else
			{
				requests[n_requests++] = MPI_REQUEST_NULL;
				completed++;
				requests[n_requests++] = MPI_REQUEST_NULL;
				completed++;
			}
		}
	}

	while (completed != n_requests)
	{
		int index = -1;

		MPI_Waitany(n_requests, requests, &index, MPI_STATUS_IGNORE);

		if (index < 0)
		{
			LogWrite("UpdateShadow::MPI_Waitany failed");
			continue;
		}

		completed++;

		if (index % 2 == 1)
		{
			int recv_dim = index / 4;
			int dir = (index % 4) / 2;

			size_t dim_a = (recv_dim + 1) % 3;
			size_t dim_b = (recv_dim + 2) % 3;

			if (dim_b < dim_a)
				swap(dim_a, dim_b);

			auto copy_idx = dir == 0 ? 0 : sz[recv_dim] - 1;

			CopyEdgeFromShadowBuf(data, recv_buffers[recv_dim][dir], copy_idx,
			                      sz[dim_a], sz[dim_b], sz[recv_dim],
			                      stride[dim_a], stride[dim_b], stride[recv_dim]);
		}
	}


	LogWrite("   shadow updated");
}

void ComputeError(size_t t)
{
	// not really needed, sync here for honest estimation of iteration time
	MPI_Barrier(cart);
	LogWrite("Fill analytical solution...");

	FillAnalytical(computed_data[t % 2], s_task_params.tau * (t - 1), s[0], s[1], s[2],
	               sz[0], sz[1], sz[2], stride[0], stride[1]);

	LogWrite("Compute error...");

	auto local_error = MaxAbsoluteError(computed_data[(t - 1) % 2], computed_data[t % 2], sz[0], sz[1], sz[2], stride[0],
	                                    stride[1]);

	double glob_error = 0;

	MPI_Reduce(&local_error, &glob_error, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

	LogWrite("    max absolute error: " + to_string(glob_error));

}

void PrintData(double *data, int zdim, int zproc)
{
	fflush(stdout);
	MPI_Barrier(cart);
	for (int iproc = 0; iproc < dim_procs[0]; iproc++)
	{
		for (int jproc = 0; jproc < dim_procs[1]; jproc++)
		{
			if (cart_coords[0] == iproc && cart_coords[1] == jproc && cart_coords[2] == zproc)
			{
				printf("%d %d %d\n", cart_coords[0], cart_coords[1], cart_coords[2]);
				for (int line = 0; line < sz[0]; line++)
				{
					for (int c = 0; c < sz[1]; c++)
						printf("%+0.6lf ", data[line * stride[0] + c * stride[1] + zdim * stride[2]]);

					printf("\n");
				}
			}

			fflush(stdout);
			usleep(10000);
			MPI_Barrier(cart);
		}
	}
}

int main(int argc, char **argv)
{
	Init(argc, argv);

	size_t t = 0;
	double ts = 0;

	LogWrite("Fill u0, u1...");

	for (; t < 2; t++)
	{
		ts = s_task_params.tau * t;
		LogWrite("    t = " + to_string(t) + ", ts = " + to_string(ts));
		FillAnalytical(computed_data[t % 2], ts, s[0], s[1], s[2],
		               sz[0], sz[1], sz[2], stride[0], stride[1]);
	}

	LogWrite("Fill done, iterating...");
	for (; t < s_task_params.K; t++)
	{
		ts = s_task_params.tau * t;
		LogWrite("    t = " + to_string(t) + ", ts = " + to_string(ts));

		auto *current = computed_data[t % 2];
		auto *prev = computed_data[(t - 1) % 2];

		UpdateShadow(prev);

#pragma omp parallel for collapse(3) shared(sz, stride) firstprivate(current, prev) default(none)
		for (size_t i = 1; i < sz[0] - 1; i++)
			for (size_t j = 1; j < sz[1] - 1; j++)
				for (size_t k = 1; k < sz[2] - 1; k++)
					MakeStep(current, prev, i, j, k, stride[0], stride[1]);
	}

	ComputeError(t);

	Finalize();
}