# HyDE-DF implementation

import numpy as np

import cython

from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count




def HyDE_DF(deParameters, otherParameters, initialSolution):

    # Generate population
    def genpop(a, b, lowMatrix, upMatrix, solution):
        temp = np.random.uniform(low=lowMatrix, high=upMatrix, size=(a, b))

        return temp

    # Trial generation
    def generate_trial(F_weight, F_CR, FM_pop, FVr_bestmemit, I_NP, I_D, FVr_rot, linear_decrease):

        # Save the old population
        FM_popold = FM_pop

        # Index pointer array
        FVr_ind = np.random.permutation(5)

        # Shuffle locations of vectors
        FVr_a1 = np.random.permutation(I_NP)

        # Rotate indices by ind[0] positions
        FVr_rt = (FVr_rot + FVr_ind[0]) % I_NP

        # Rotate vector locations
        FVr_a2 = FVr_a1[FVr_rt]

        # Shuffled populations
        FM_pm1 = FM_popold[FVr_a1, :]
        FM_pm2 = FM_popold[FVr_a2, :]

        FM_mpo = None

        # Meaning the same F_CR for all individuals
        if len(F_CR) == 1:
            # All random numbers < F_CR are 1, 0 otherwise
            #FM_mui = (np.random.normal(size=(I_NP, I_D)) < F_CR).astype(int)
            FM_mui = (np.random.uniform(size=(I_NP, I_D)) < F_CR).astype(int)

            # Inverse mask to FM_mui
            FM_mpo = np.logical_not(FM_mui).astype(int)

        # Meaning a different F_CR for each individual
        else:
            # All random numbers < F_CR are 1, 0 otherwise
            #FM_mui = (np.random.normal(size=(I_NP, I_D)) < F_CR).astype(int)
            FM_mui = (np.random.uniform(size=(I_NP, I_D)) < F_CR).astype(int)

            # Inverse mask to FM_mui
            FM_mpo = np.logical_not(FM_mui).astype(int)


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        FM_bm = np.tile(FVr_bestmemit, (I_NP, 1))

        # Exponential decreasing function
        ginv = np.exp(1-(1/linear_decrease**2))

        #differential variation
        repmat0 = np.reshape(np.tile(F_weight[:, 2], (1, I_D)), (F_weight.shape[0], I_D))
        repmat1 = np.reshape(np.tile(F_weight[:, 0], (1, I_D)), (F_weight.shape[0], I_D))
        repmat2 = np.reshape(np.tile(F_weight[:, 1], (1, I_D)), (F_weight.shape[0], I_D))

        diff_var = ginv * (repmat1 * (FM_bm * (repmat2 + np.random.uniform(size=(I_NP, I_D)) - FM_popold)))

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        FM_ui = FM_popold + repmat0 * (FM_pm1 - FM_pm2) + diff_var

        FM_ui = FM_popold * FM_mpo + FM_ui * FM_mui
        FM_base = FM_bm

        return FM_ui, FM_base, None

    # Update aux function
    def _update(p, lowMatrix, upMatrix, BRM, FM_base):
        if BRM == 1: # Our method
            p = np.clip(p, lowMatrix, upMatrix)

        elif BRM == 2: # Random initialization - DOES NOT WORK
            idx = [np.where(p < lowMatrix), np.where(p > upMatrix)]
            replace = np.random.uniform(low=lowMatrix[idx[0][0], idx[0][1]],
                                        high=upMatrix[idx[1][0], idx[1][1]],
                                        size=(len(idx), 1))
            p[idx] = replace
        return p


    #-----This is just for notational convenience and to keep the code uncluttered.--------
    I_NP = deParameters.I_NP
    F_weight = deParameters.F_weight
    F_CR = deParameters.F_CR
    I_D = otherParameters.dim     #Number of variables or dimension
    deParameters.nVariables = I_D

    FVr_minbound = otherParameters.lowerlimit
    FVr_maxbound = otherParameters.upperlimit

    I_itermax: cython.int = deParameters.I_itermax


    #Repair boundary method employed
    BRM: cython.int = deParameters.I_bnd_constr     #1: bring the value to bound violated
    #2: repair in the allowed range
    #3: Bounce-back

    #-----Check input variables---------------------------------------------
    if I_NP < 5:
        I_NP = 5
        print('I_NP increased to minimal value 5\n')

    if (F_CR < 0) | (F_CR > 1):
        F_CR = 0.5
        print('F_CR should be from interval [0, 1] - set to default value 0.5\n')

    if I_itermax <= 0:
        I_itermax = 200
        print('I_itermax should be > 0 - set to default value 200\n')


    #-----Initialize population and some arrays-------------------------------
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # pre-allocation of loop variables
    fitMaxVector = np.empty((1, I_itermax))
    fitMaxVector[:] = np.nan

    # limit iterations by threshold
    #gen = 0; #iterations


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #----FM_pop is a matrix of size I_NPx(I_D+1). It will be initialized------
    #----with random values between the min and max values of the-------------
    #----parameters-----------------------------------------------------------
    # FLC modification - vectorization
    minPositionsMatrix = np.tile(FVr_minbound, (I_NP, 1))
    maxPositionsMatrix = np.tile(FVr_maxbound, (I_NP, 1))
    deParameters.minPositionsMatrix = minPositionsMatrix
    deParameters.maxPositionsMatrix = maxPositionsMatrix

    # generate initial population.
    FM_pop = genpop(I_NP, I_D, minPositionsMatrix, maxPositionsMatrix,
                    otherParameters.initial_solution)

    FM_aux = [EnergyCommunity(parameters=otherParameters.param_dictionary,
                              variables=otherParameters.decoded_initialSolution) for i in range(I_NP)]

    #If you want to inject initial solutions
    if otherParameters.initial_solution is not None:
        FM_aux[0].doIteration()
        FM_pop[0, :] = otherParameters.initial_solution
        for solution_idx in range(1, len(FM_aux)):
            FM_aux[solution_idx].newIteration(FM_pop[solution_idx, :])
    else:
        for solution_idx in range(len(FM_aux)):
            FM_aux[solution_idx].newIteration(FM_pop[solution_idx, :])

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #------Evaluate the best member after initialization----------------------
    # Modified by EG
    S_val = np.array([temp_obj.objFn for temp_obj in FM_aux])
    S_val = S_val.ravel()
    I_best_index = np.argmin(S_val) # This mean that the best individual correspond to the best worst performance
    FVr_bestmemit = FM_pop[I_best_index, :] # best member of current iteration

    fitMaxVector[:, 0] = S_val[I_best_index] #We save the mean value and mean penalty value

    # The user can decide to save the mean, best, or any other value here
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #------DE-Minimization------------------------------------------------
    #------FM_popold is the population which has to compete. It is--------
    #------static through one iteration. FM_pop is the newly--------------
    #------emerging population.-------------------------------------------
    FVr_rot  = np.arange(I_NP)             # rotating index array (size I_NP)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% HYDE self-adaptive parameters
    F_weight_old = np.tile(F_weight, (I_NP, 3))
    F_weight = F_weight_old.copy()
    F_CR_old = np.tile(F_CR, (I_NP, 1))
    F_CR = F_CR_old.copy()

    best_instance = None

    # Task for multiprocessing
    def new_iteration(candidate):
        FM_aux[candidate].newIteration(FM_ui[candidate, :])
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    itermax_range: cython.float[I_itermax] = np.arange(I_itermax)

    # Initialize previous best value with a large value
    prev_best_val = 1e10
    current_tolerance = 0
    lin_decr: cython.float = (I_itermax) / I_itermax
    for gen in tqdm.tqdm(itermax_range):
        #% Calculate decay function factor a = itr / MaxItr;
        lin_decr = (I_itermax - gen) / I_itermax

        # Update HyDE-DF values
        ind1 = np.random.uniform(size=(I_NP, 3)) < 0.1
        ind2 = np.random.uniform(size=(I_NP, 1)) < 0.1

        F_weight[ind1] = (0.1 + np.random.uniform(size=(I_NP, 3)) * 0.9)[ind1]
        F_weight[~ind1] = F_weight_old[~ind1]

        F_CR[ind2] = np.random.uniform(size=(I_NP, 1))[ind2]
        F_CR[~ind2] = F_CR_old[~ind2]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        FM_ui, FM_base, _ = generate_trial(F_weight, F_CR, FM_pop, FVr_bestmemit, I_NP, I_D, FVr_rot, lin_decr)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ## Boundary Control
        FM_ui = _update(FM_ui, minPositionsMatrix, maxPositionsMatrix, BRM, FM_base)

        #for candidate in np.arange(I_NP):
        #    FM_aux[candidate].newIteration(FM_ui[candidate, :])


        if otherParameters.use_parallel:
            # Use multiprocessing for the newIteration calling
            with ThreadPool(processes=cpu_count()) as pool:
                pool.map(new_iteration, np.arange(I_NP))
        else:
            # Use single thread for the newIteration calling
            for candidate in np.arange(I_NP):
                FM_aux[candidate].newIteration(FM_ui[candidate, :])

        S_val_temp = np.array([temp_obj.objFn for temp_obj in FM_aux]).ravel()
        #S_val_temp = S_val_temp.ravel()

        # Elitist Selection
        ind = np.where(S_val_temp < S_val)
        S_val[ind] = S_val_temp[ind]
        FM_pop[ind, :] = np.array([temp_obj.encoded for temp_obj in FM_aux])[ind]


        # Update best results
        S_bestval = np.min(S_val)
        I_best_index = np.argmin(S_val)
        best_instance = FM_aux[I_best_index]
        FVr_bestmemit = FM_pop[I_best_index, :]

        # Save best parameters (similar to jDE)
        F_weight_old[ind, :] = F_weight[ind, :]
        F_CR_old[ind] = F_CR[ind]

        ## Store fitness evolution and obj fun evolution as well
        fitMaxVector[:, gen] = S_bestval

        # Check if the difference between the current best value and the previous best value is less than epsilon
        if abs(S_bestval - prev_best_val) < deParameters.epsilon:
            if current_tolerance < deParameters.tolerance:
                current_tolerance += 1
            else:
                break
        else:
            current_tolerance = 0
            # If the difference is less than epsilon, stop the algorithm and return the current best solution
            # break

        # Update previous best value
        prev_best_val = S_bestval

    Fit_and_p = fitMaxVector[0, gen]

    return Fit_and_p, FVr_bestmemit, fitMaxVector, best_instance