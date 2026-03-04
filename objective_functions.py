from wamit_utils import *
from mesh_axisymmetric_shape import *
import numpy as np

def geometry_layout_func(vector, Hs:float, Tp:float, h:float, verbose:bool=False)->float:
    writer = WamitInputGenerator()
    writer.input_init()
    writer.pot_file(vector, h)
    generate_cylinder_hemisphere_mesh(vector[6], vector[7], '.\\wamit_optimization\\wec.gdf', False)
    run_wamit(verbose=verbose)
    parser = WamitOutputParser()
    P_inc = parser.incident_power(Hs=Hs, Tp=Tp, gamma=1.4, h=h)
    total_power, individual_powers = parser.calc_power(Hs=Hs, Tp=Tp, gamma=1.4)
    # individual_CWR = individual_powers / (P_inc * 2 * vector[6])
    # CWR = np.average(individual_CWR)
    # return CWR, individual_CWR
    return total_power, individual_powers

if __name__ == "__main__":
    # vector = np.array([6.7, 27.8, 14.7, 27.7, 10.4, 0.0, 1.2, 5.0])
    vector = np.array([12.7, 30.0, 3.6, 30.0, 7.8, 0.0, 2.5, 4.8])
    # generate_cylinder_hemisphere_mesh(vector[6], vector[7], '.\\wamit_optimization\\wec.gdf', False)
    total_power, individual_powers = geometry_layout_func(vector, Hs=1.07, Tp=5.46, h=50.0, verbose=False)
    print(f"Calculated total power: {total_power:.4f}")
    print(f"Individual powers: {individual_powers}")