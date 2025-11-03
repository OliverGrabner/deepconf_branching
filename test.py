from math import ceil

# Variables
start_traces = 16 # how many traces we start with
max_traces = 512 # what is the maximum allowed traces we can have?
selected_percent = 0.60 # top % considered at each stride 

# ===================================
# Constant  max_traces = start_traces + m*t
#
# m = number of stop & branches we cross 
# m = (max_traces - start_traces) / (0.75 * avg CoT length)
# 
# t = new branches created at each m stop
# t = ciel*(max_traces - start_traces) / m )
# ===================================
def branches_per_stride(avg_cot_length):
    k = 0.75 * avg_cot_length  # hypothesis: last 10-20 percent of CoT is verification / no new ideas
    t = ceil((max_traces-start_traces) / k) # how many branches we make 
    return t
