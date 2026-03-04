import numpy as np
import time
import inspect
import utils

def DE(func, bounds, particle_size:int, loc_name:str, h:float, Hs:float, Tp:float,
       step_size:np.ndarray, max_iter=100, F=0.7, CR=0.5,
       results_path='geo_layout_cal_results.res', iter_path='geo_layout_iter_results.res', verbose=False):
    algorithm = 'DE'
    func_name = inspect.stack()[0].function
    memory_list = []

    lower = np.array([bounds[key][0] for key in bounds])
    upper = np.array([bounds[key][1] for key in bounds])
    dimension = len(bounds)

    pops = []
    while len(pops) < particle_size:
        raw_cand = np.random.uniform(lower, upper, dimension)

        cand = utils.apply_step_size(raw_cand, step_size)
        cand = np.clip(cand, lower, upper)

        if utils.distance_check(cand):
            pops.append(cand)
    
    pops = np.array(pops)
    
    fitness_list = []
    for idx, vector in enumerate(pops):
        utils.print_start_message(idx, vector)
        start_eval_time = time.time()
        fitness, each_wec_CWR = func(vector, Hs, Tp, h, verbose=verbose)
        elapsed = time.time() - start_eval_time

        utils.print_eval_message(elapsed)
        fitness_list.append(fitness)
        memory_list.append(list(vector) + [fitness])
        utils.write_results(list(vector) + [fitness] + list(np.round(each_wec_CWR, 3)), results_path)

    best_idx = np.argmax(fitness_list)
    gbest = pops[best_idx].copy()
    gbest_fitness = fitness_list[best_idx]

    utils.print_iter_start_message(func_name, loc_name, Hs, Tp, h)

    for iteration in range(1, max_iter + 1):
        print(f'\n Iteration {iteration} Calculate...')
        cnt_memory, iter_start = 0, time.time()
        for i in range(particle_size):
            print("-"*50)
            print(f'      🔄 Updating Particle {i + 1}/{particle_size}')
            while True:
                idxs = [idx for idx in range(particle_size) if idx != i]
                a, b, c = pops[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lower, upper)
                cross_points = np.random.rand(dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, pops[i])
                trial = utils.apply_step_size(trial, step_size)
                trial = np.clip(trial, lower, upper)
                if utils.distance_check(trial):
                    break

            trial = utils.normalize_layout_vector(trial)

            memory = next((item for item in memory_list if np.allclose(np.array(item[:-1]), trial, atol=1e-4)), None)

            if memory is None:
                start_eval_time = time.time()
                trial_fitness, each_wec_CWR = func(trial, Hs, Tp, h, verbose=verbose)
                elapsed = time.time() - start_eval_time
                utils.print_eval_message(elapsed)
                memory_list.append(list(trial) + [trial_fitness])
                utils.write_results(list(trial) + [trial_fitness] + list(np.round(each_wec_CWR, 3)), results_path)
            else:
                trial_fitness = memory[-1]
                cnt_memory += 1
                print(f" Memory Hit! Fitness: {trial_fitness:.3f}")

            if trial_fitness > fitness_list[i]:
                pops[i], fitness_list[i] = trial, trial_fitness
                if trial_fitness > gbest_fitness:
                    gbest, gbest_fitness = trial.copy(), trial_fitness
                    print(f' Global best Updated -> {gbest_fitness:.3f}')

        total_time = np.round((time.time() - iter_start) / 60, 2)
        utils.write_results(list(gbest) + [gbest_fitness, total_time], iter_path)
        utils.print_summary_message(iteration, gbest, gbest_fitness, total_time, cnt_memory)
    utils.move_results_file(algorithm, loc_name)