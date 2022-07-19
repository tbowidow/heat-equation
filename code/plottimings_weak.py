from numpy import *
from matplotlib import pyplot
import re
import sys

def read_file(file_path):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines

def find_thread_counts(lines):
    ind = []
    for i, l in enumerate(lines):
        if "Thread" in l:
            ind.append(i)
    return ind

def break_lines(lines, ind):
    out = []
    for i in range(len(ind) - 1):
        out.append(lines[ind[i]:ind[i + 1]])
    out.append(lines[ind[len(ind) - 1]:])
    return out

def map_threads_to_lines(broken_lines):
    out = {}
    for group in broken_lines:
        first = group[0]
        # regex attributed to StackOverflow
        nums = re.findall(r'\d+', first)
        num = int(nums[0])
        out[num] = group
    return out

def get_timings_per_thread(map_t_to_l):
    out = {}
    for th, li in map_t_to_l.items():
        subout = []
        for line in li:
            if "real\t" in line:
                # regex attributed to StackOverflow
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                secs = float(nums[1]) + (60.0 * float(nums[0]))
                subout.append(secs)
        out[th] = subout
    return out

def average_timings(timings):
    out = {}
    for th, li in timings.items():
        sum_timings = 0.0
        for time_s in li:
            sum_timings = sum_timings + time_s
        out[th] = sum_timings / float(len(li))
    return out

def plot_timings(avg_timings):
    x = zeros((len(avg_timings),))
    y = zeros((len(avg_timings),))
    i = 0
    for th, fl in avg_timings.items():
        x[i] = float(th)
        y[i] = fl
        i = i + 1
    
    pyplot.figure(1)
    
    ref = 4*x
    pyplot.loglog(x,y, marker='o')
    pyplot.loglog(x,ref, marker='o')

    pyplot.legend(["Numerical Timings","Reference Line: Factor of 4 increase"], 
                  fontsize='large')
    pyplot.title('Weak Scaling Timing', fontsize="large")
    pyplot.xlabel('Workers', fontsize='large')
    pyplot.ylabel('Time (s)', fontsize='large')

    pyplot.savefig('weakscalingtiming.png', dpi=500, format='png')

fpath = sys.argv[1]
lines = read_file(fpath)
threads_i = find_thread_counts(lines)
broken_lines = break_lines(lines, threads_i)
map_t_to_l = map_threads_to_lines(broken_lines)
timings = get_timings_per_thread(map_t_to_l)
avg_timings = average_timings(timings)
plot_timings(avg_timings)
