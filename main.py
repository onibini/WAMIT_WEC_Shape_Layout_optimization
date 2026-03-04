import config
from optimizers import DE
import os
from objective_functions import geometry_layout_func

if __name__ == "__main__":
    if os.path.exists(config.RESULTS_PATH):
        os.remove(config.RESULTS_PATH)
    if os.path.exists(config.ITER_PATH):
        os.remove(config.ITER_PATH)

    DE(geometry_layout_func, config.BOUNDS, config.PARTICLE_SIZE, config.CURRENT_LOC,
       config.h, config.Hs, config.Tp, config.STEP_SIZE, config.MAX_ITER,
       config.DE_PARAMS['F'], config.DE_PARAMS['CR'], config.RESULTS_PATH, config.ITER_PATH, verbose=False)