# This is our main file

input_folder = "testcases/"
output_folder = "output/"

# INPUT VARIABLES
baseline_cost, action_cost = 0, 0
CPU_cost, MEM_cost = 0, 0 # cloud costs
B, CPU, MEM, ACC, cost_per_set = 0, 0, 0, 0, 0 # BBU profile
num_slices, time_horizon, CPU_ACC_ratio = 0, 0, 0
# the next will be repeated for each slice s (num_slices repeats)
cpu, mem, acc = 0, 1, 2
l_a, l_b, l_c, l_d = 0, 1, 2, 3
CU = {} # CU resource unit
DU = {} # DU resource unit
PHY = {} # PHY resource unit
L = {} # IO cost
T = {} # T traffic unit

# OUTPUT VARIABLES
solution_matrix = {} # solution matrix
cloud_cost = [] # cloud cost
BBU_cost = [] # BBU cost
IO_cost = [] # IO cost
OPEX = 0 # OPEX
score = 0 # score
execution_time = 0 # execution time

# OTHER VARIABLES
""" 
    for example deployed[0] = {'cu': [0, 1], 'du': [0, 1], 'phy': [0, 1]} means that at time 0 the service was BBB, at time 1 it was CCC 
    1 for cloud, 0 for BBU
"""
deployed = {} # deployed service to keep track of the deployed services over time

"""
    Parse the input file and store the data in the variables
    @param file_name: the name of the input file
"""
def parse_init(file_name):
    global baseline_cost, action_cost, CPU_cost, MEM_cost
    global B, CPU, MEM, ACC, cost_per_set
    global num_slices, time_horizon, CPU_ACC_ratio, CU, DU, PHY, L_a, L_b, L_c, L_d, T

    with open(input_folder + file_name) as f:
        baseline_cost, action_cost = map(int, f.readline().split())
        CPU_cost, MEM_cost = map(int, f.readline().split())
        B, CPU, MEM, ACC, cost_per_set = map(int, f.readline().split())
        num_slices, time_horizon, CPU_ACC_ratio = map(int, f.readline().split())
        
        for i in range(num_slices):
            CU[i] = list(map(int, f.readline().split()))
            DU[i] = list(map(int, f.readline().split()))
            PHY[i] = list(map(int, f.readline().split()))
            L[i] = list(map(int, f.readline().split()))
            T[i] = list(map(int, f.readline().split()))

"""
    Print the output in the output file
"""
def print_output():
    pass

def cloud_cost_compute(CPU, MEM, ACC, T):
    CPU_req = CPU + X * ACC
    return CPU_req * T * CPU_cost + MEM * T * MEM_cost

def cloud_cost(s, t, cu=false, du=false, phy=false):
    cloud_cost = 0

    if cu:
        cloud_cost += cloud_cost_compute(CU[s][cpu], CU[s][mem], CU[s][acc], T[s][t])
    if du:
        cloud_cost += cloud_cost_compute(DU[s][cpu], DU[s][mem], DU[s][acc], T[s][t])
    if phy:
        cloud_cost += cloud_cost_compute(PHY[s][cpu], PHY[s][mem], PHY[s][acc], T[s][t])

    return cloud_cost

def BBU_cost_compute(CPU_allocated_tot, MEM_allocated_tot, ACC_allocated_tot):
    return max(math.ceil(CPU_allocated_tot / CPU), math.ceil(MEM_allocated_tot / MEM), math.ceil(ACC_allocated_tot / ACC)) * cost_per_set

def BBU_cost(s, cu=false, du=false, phy=false):
    CPU_allocated, MEM_allocated, ACC_allocated = 0, 0, 0

    if cu:
        CPU_allocated += CU[s][cpu]
        MEM_allocated += CU[s][mem]
        ACC_allocated += CU[s][acc]
    if du:
        CPU_allocated += DU[s][cpu]
        MEM_allocated += DU[s][mem]
        ACC_allocated += DU[s][acc]
    if phy:
        CPU_allocated += PHY[s][cpu]
        MEM_allocated += PHY[s][mem]
        ACC_allocated += PHY[s][acc]

    return CPU_allocated, MEM_allocated, ACC_allocated

"""
    Compute the IO cost
    @param s: the slice
    @param cu: CU resource unit
    @param du: DU resource unit
    @param phy: PHY resource unit
    Here true means that the resource unit is allocated on cloud, false otherwise
    @return IO_cost: the IO cost
"""
def IO_cost(s, cu=false, du=false, phy=false):
    IO_cost = 0

    if cu:
        if du:
            if phy:
                IO_cost = L[s][l_d]
            else:
                IO_cost = L[s][l_c]
        else:
            IO_cost = L[s][l_b]
    else:
        IO_cost = L[s][l_a]

    return IO_cost * T[s][t]

def action_cost(s, t):
    service_reloc = 0

    # check for relocation from deployed dict
    if deployed[s]['cu'][t] != deployed[s]['cu'][t-1]:
        service_reloc += 1
    if deployed[s]['du'][t] != deployed[s]['du'][t-1]:
        service_reloc += 1
    if deployed[s]['phy'][t] != deployed[s]['phy'][t-1]:
        service_reloc += 1

    return service_reloc * action_cost

if __name__ == "__main__":

    parse_init("toy_example.txt")

    # print(baseline_cost, action_cost)
    # print(CPU_cost, MEM_cost)
    # print(B, CPU, MEM, ACC, cost_per_set)
    # print(num_slices, time_horizon, CPU_ACC_ratio)
    # for i in range(num_slices):
    #     print(CU[i])
    #     print(DU[i])
    #     print(PHY[i])
    #     print(L[i])
    #     print(T[i])

    # initialize deployed dic
    for s in range(num_slices):
        deployed[s] = {'cu': [], 'du': [], 'phy': []}

    print_output()
    