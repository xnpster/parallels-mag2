import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = './omp_tests'


def extract_metrics(file_path):
    fetch_timings = {
        'Fill u0, u1...' : 'fill',
        'Fill done, iterating...' : 'iter',
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
            
            if 'omp num threads:' in ln:
                params['t'] = int(ln[ln.find(':') + 1:])

            if 'max absolute error' in ln:
                params['resid'] = float(ln[ln.find(':') + 1:])
    
    result = {
        'fill': timings['iter'] - timings['fill'],
        'iter': timings['error'] - timings['iter'],
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

print(df.sort_values(['K', 'N', 't']))

fig, axs_2d = plt.subplots(2, 2)

axs = axs_2d.flat
axs_idx = 0

for group_key, group in df.groupby(['K', 'N']):
    group = group.sort_values('t')
    serial = group[group['t'] == 1]

    ticks = group['t'].astype(str).values

    ax = axs[axs_idx]
    axs_idx += 1

    for section in ['fill', 'iter', 'error']:
        ax.plot(ticks,
                 serial[section].values / group[section], 
                 'v--',
                 label=section,
                 linewidth=2
                 )
    
    ax.set(xlabel = 'OpenMP threads, num', ylabel = 'Acceleration, times')

    k, n = group_key

    ax.set_title(f'K: {k}, N: {n}')

    ax.set_xticks(ticks)
    ax.set_yticks(np.arange(1, 8))
    ax.grid()
    ax.legend()

plt.gcf().set_size_inches(10, 10)
plt.savefig(RESULTS_DIR + '/combined.png', bbox_inches='tight')
plt.close()