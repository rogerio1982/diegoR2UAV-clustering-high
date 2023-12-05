# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:26:50 2023

@author: dl_ca
"""

moo_nobj = 4
verbose  = 1
runtime  = 60

num_evals = 100
size_pop  = int(np.max([5, int(np.ceil(np.sqrt(num_evals)))]))
num_gen   = int(np.ceil(num_evals/np.sqrt(num_evals)))

space_min = [2]# + list(data.min())
space_max = [int(data.shape[0]/2)]# + list(data.max())

print(num_evals, size_pop, num_gen)
print(space_min, space_max)
print("\nNum. objectives:", moo_nobj)
print("Num. input var.:", len(space_min))

# Prepare simulation
estimator = dict()
estimator["max_time_model"] = runtime
estimator["moo_temp_"]      = []
estimator["ndf_temp_"]      = []
estimator["opt_memory_"]    = []

# Define problem
problem = pg.problem(ml_function_MOO(space_min, space_max, moo_nobj)) 
if verbose >= 1:
    print("\n%s" % ('-----'*20))
    print(problem)

# Define population
size_pop = validate_size_pop(moo_nobj, size_pop, verbose)
# try:
model_time_init = time.perf_counter() 
pop = pg.population(prob = problem, size = size_pop) # The initial population
# except Exception as e:
#     print(e)
#     pass
if verbose >= 3:
    try:
        ax = pg.plot_non_dominated_fronts(pop.get_f()) # show plot
    except: pass
print("\n%s" % ('-----'*20))

# Define BIA
bio_alg = pg.algorithm(pg.moead(gen = num_gen, neighbours=min(int(size_pop/2), 20))) # (pg.moead(gen=250)) #
# bio_alg = pg.algorithm(pg.nsga2(gen = num_gen)) # (pg.nsga2(gen=20)) #

bio_alg.set_verbosity(max(int(verbose), 1)) # Warning: higher values causes averaging of more values
if verbose >= 1:
    print("\n%s" % ('-----'*20))
    print(bio_alg)
    print()

# Perform optimization
# try:
model_time_init = time.perf_counter() 
pop = bio_alg.evolve(pop) # The actual optimization process
# except Exception as e:
#     print(e)
#     pass
if verbose >= 3:
    print("Inputs:\n", pop.get_x()) # np.round(pop.get_x()).astype(int)
    print("Outputs:\n", pop.get_f())
    ndf = pg.fast_non_dominated_sorting(pop.get_f())[0]
    print("Non-dominated candidates:", len(ndf[0]), "\n", ndf)                    
    try:
        ax = pg.plot_non_dominated_fronts(pop.get_f()) # show plot
    except: pass
print("\n%s" % ('-----'*20))

# Get scores after optimization
try:
    ndf_df_temp = pd.DataFrame(estimator["ndf_temp_"])
    ndf_df_temp = ndf_df_temp.drop_duplicates()

    ndf_df_temp_f = ndf_df_temp.values[:, len(space_min):] # !
    ndf_temp_f    = pg.fast_non_dominated_sorting(ndf_df_temp_f)[0]

    ndf_temp_f1 = ndf_temp_f[0]
    ndf_temp_f1 = list(ndf_df_temp.index[ndf_temp_f1])

#     moo_df_temp = pd.DataFrame(estimator["moo_temp_"], columns=["s_1", "s_2"])        
    moo_df_temp = pd.DataFrame(estimator["ndf_temp_"], columns=["solution", "s_1", "s_2", "s_3", "s_4"])        
    moo_df_temp = moo_df_temp.iloc[ndf_temp_f1]
except:
    raise Exception()
    moo_df_temp = pd.DataFrame(estimator["moo_temp_"], columns=["s_1", "s_2", "s_3", "s_4"])        
    moo_df_temp = moo_df_temp.drop_duplicates()
print("Sorting based on {} objectives".format(ndf_df_temp.shape[1]-len(space_min)))
moo_df_temp