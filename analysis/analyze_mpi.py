import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = './mpi_tests'


def extract_metrics(file_path):
    fetch_timings = {
        'Fill u0, u1...' : 'fill',
        'Fill done, iterating...' : 'iter',
        'Fill analytical solution...' : 'iter_done',
        'Compute error...': 'error',
        'max absolute error:' : 'done'
    }

    timings = {}
    params = {}

    with open(file_path) as f:
        for ln in f:
            for k, v in fetch_timings.items():
                if k in ln:
                    i1 = ln.find('[')
                    i2 = ln.find(']')

                    timestamp = float(ln[i1+1:i2])
                    timings[v] = timestamp
            
            if 'K:' in ln:
                params['K'] = int(ln[ln.find(':') + 1:])
            
            if 'N:' in ln:
                params['N'] = int(ln[ln.find(':') + 1:])

            if 'P:' in ln:
                params['P'] = int(ln[ln.find(':') + 1:])

            if 'Lx:' in ln:
                params['L'] = str(ln[ln.find(':') + 2:ln.find(',')])
            
            if 'omp num threads:' in ln:
                params['t'] = int(ln[ln.find(':') + 1:])

            if 'max absolute error' in ln:
                params['resid'] = float(ln[ln.find(':') + 1:])
    
    result = {
        'fill': timings['iter'] - timings['fill'],
        'iter': timings['iter_done'] - timings['iter'],
        'error': timings['done'] - timings['error'],
    }

    result.update(params)

    return result


row_list = []
for file_name in os.listdir(RESULTS_DIR):
    if '.log' not in file_name:
        continue

    row = extract_metrics(RESULTS_DIR + '/' + file_name)
    row_list.append(row)

df = pd.DataFrame(row_list)

sort_keys = ['K', 'N', 'L', 'P']
display_keys = ['K', 'N', 'L', 'P', 'fill', 'iter', 'error', 'resid']



print('\\begin{tabular}{|l|l|l|l|l|l|l|l|}')
print('\\hline')
print('K & N & L & MPI-процессы & fill, сек & iter, сек & error, сек & невязка \\\\')

idx = 0
for i, r in df.sort_values(sort_keys)[display_keys].iterrows():
    if idx % 9 == 0:
        print('\\hline')

    print(' & '.join((f'{e:0.6f}' if isinstance(e, float) else str(e) ) for e in r), '\\\\')

    idx += 1
print('\\hline')
print('\\end{tabular}')

fig, axs_2d = plt.subplots(1, 1)

# axs = axs_2d.flat
axs = [axs_2d]

axs_idx = 0

group_by_keys = ['K', 'N', 'L']

for group_key, group in df.groupby(group_by_keys):
    group = group.sort_values('P')
    serial = group[group['P'] == 1]

    ticks = group['P'].astype(str).values

    ax = axs[axs_idx]
    axs_idx += 1

    for section in ['fill', 'iter', 'error']:
        ax.plot(ticks,
                 serial[section].values / group[section], 
                 'v--',
                 label=section,
                 linewidth=2
                 )
    
    ax.set(xlabel = 'MPI processes', ylabel = 'Acceleration, times')

    title = ''
    for i in range(len(group_by_keys)):
        title += f'{group_by_keys[i]}: {group_key[i]} '

    ax.set_title(title)

    ax.set_xticks(ticks)
    # ax.set_yticks(np.arange(1, 8))
    ax.grid()
    ax.legend()

plt.gcf().set_size_inches(10, 10)
plt.savefig(RESULTS_DIR + '/mpi.png', bbox_inches='tight')
plt.close()