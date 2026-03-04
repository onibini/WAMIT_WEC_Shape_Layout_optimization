import numpy as np
import os
import textwrap
import psutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy.integrate import simpson
from scipy.optimize import newton

def JONSWAP_SPEC(omega:np.ndarray, Hs:float, Tp:float, gamma:float)->np.ndarray:
    wp = 2 * np.pi / Tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-((omega - wp)**2) / (2 * sigma**2 * wp**2))

    beta_denominator = 0.23 + 0.0336 * gamma - 0.185 * (1.9 + gamma)**-1
    beta = (0.0624 / beta_denominator) * (1.094 - 0.01915 * np.log(gamma))

    S = np.zeros_like(omega)
    non_zero_indices = omega > 0

    omega_safe = omega[non_zero_indices]
    r_safe = r[non_zero_indices]

    term1 = beta * (Hs ** 2 * wp ** 4) / (omega_safe ** 5)
    term2 = np.exp(-1.25 * (omega_safe / wp) ** -4)
    term3 = gamma ** r_safe
    S[non_zero_indices] = term1 * term2 * term3

    return S

def remove_file():
    remove_list = ['wec.1', 'wec.2', 'wec.3', 'wec.4',
                   'wec.out', 'wec.p2f']
    for file in remove_list:
        if os.path.exists(os.path.join(r'wamit_optimization', file)):
            os.remove(os.path.join(r'wamit_optimization', file))
    print("    🗑️ Previous WAMIT output files removed.")

def run_wamit(verbose=False):
    remove_file()
    if verbose:
        with subprocess.Popen([r'C:/WAMITv7/wamit.exe', 'fnames.wam'], cwd=r'wamit_optimization') as proc:
            proc.wait()
    else:
        with subprocess.Popen([r'C:/WAMITv7/wamit.exe', 'fnames.wam'], cwd=r'wamit_optimization',
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
            proc.wait()

class WamitInputGenerator:
    def __init__(self, base_dir=r'wamit_optimization', project_title='Geometry_Layout Optimization'):
        self.base_dir = base_dir
        self.project_title = project_title

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_path(self, filename:str)->str:
        return os.path.join(self.base_dir, filename)
    
    def input_init(self):
        self.fname_file()
        self.cfg_file()
        self.cfgwam_file()
        self.frc_file()

    def frc_file(self):
        input = textwrap.dedent(f"""\
                                {self.project_title}
                                {"1 1 1 0 0 0 0 0 0":<20} OutputFile
                                {"0":<20} VCG1
                                {"1 0 0":<20}
                                {"0 1 0":<20}
                                {"0 0 1":<20}
                                {"0":<20} VCG2
                                {"1 0 0":<20}
                                {"0 1 0":<20}
                                {"0 0 1":<20}
                                {"0":<20} VCG3
                                {"1 0 0":<20}
                                {"0 1 0":<20}
                                {"0 0 1":<20}
                                {"0":<20} VCG4
                                {"1 0 0":<20}
                                {"0 1 0":<20}
                                {"0 0 1":<20}
                                {"0":<20} VCG5
                                {"1 0 0":<20}
                                {"0 1 0":<20}
                                {"0 0 1":<20}
                                {"0":<20}NBETAH
                                {"0":<20}NFIELD
                                """)
        with open(self._get_path('wec.frc'), 'w') as f:
            f.write(input)

    def fname_file(self):
        input = textwrap.dedent(f"""\
                                {"wec.cfg"}
                                {"wec.pot"}
                                {"wec.frc"}
                                """)
        with open(self._get_path('fnames.wam'), 'w') as f:
            f.write(input)

    def cfg_file(self):
        input = textwrap.dedent(f"""\
                                {self.project_title}
                                ipltdat=1
                                NUMHDR=1
                                IPERIN=2
                                IPEROUT=2
                                IWALLX0=0
                                ISOR=0
                                ISOLVE=1
                                MAXITT=100
                                ISCATT=0
                                IALTFRC=1
                                ILOG=1
                                USERID_PATH={r"C:/WAMITv7"}
                                """)
        with open(self._get_path('wec.cfg'), 'w') as f:
            f.write(input)

    def cfgwam_file(self):
        num_cpu = os.cpu_count()
        memory_info = psutil.virtual_memory()
        total_ram = memory_info.total / (1024 ** 3)

        input = textwrap.dedent(f"""\
                                {self.project_title}
                                RAMGBMAX={round(total_ram/2)}
                                NCPU={num_cpu}
                                """)
        with open(self._get_path('config.wam'), 'w') as f:
            f.write(input)

    def pot_file(self, vector:np.ndarray, h:float):
        input = textwrap.dedent(f"""\
                                {self.project_title}
                                {h:<20} HBOT
                                {"0 0":<20} IRAD IDIFF
                                {"-51":<20} NPER
                                {"0.5 0.05":<20} PER
                                {"1":<20} NBETA
                                {"0":<20} BETA
                                {"5":<20} NBODY
                                {"wec.gdf":<20} BODY
                                {f"{vector[0]} {vector[1]} 0 0":<20} XBODY
                                {"0 0 1 0 0 0":<20} IMODE
                                {"wec.gdf":<20} BODY
                                {f"{vector[2]} {vector[3]} 0 0":<20} XBODY
                                {"0 0 1 0 0 0":<20} IMODE
                                {"wec.gdf":<20} BODY
                                {f"{vector[4]} 0 0 0":<20} XBODY
                                {"0 0 1 0 0 0":<20} IMODE
                                {"wec.gdf":<20} BODY
                                {f"{vector[2]} {-vector[3]} 0 0":<20} XBODY
                                {"0 0 1 0 0 0":<20} IMODE
                                {"wec.gdf":<20} BODY
                                {f"{vector[0]} {-vector[1]} 0 0":<20} XBODY
                                {"0 0 1 0 0 0":<20} IMODE
                                """)
        with open(self._get_path('wec.pot'), 'w') as f:
            f.write(input)


class WamitOutputParser:
    def __init__(self, base_dir=r'wamit_optimization'):
        self.base_dir = base_dir
        self.rho = 1025.0
        self.g = 9.80665

    def _get_path(self, filename:str)->str:
        return os.path.join(self.base_dir, filename)
    
    def get_mass(self) -> float:
        fpath = self._get_path('wec.out')

        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                for line in f:
                    if 'Volumes' in line:
                        vol = float(line.split()[-1])
                        return vol * self.rho
                    
    def get_stiffness(self) -> np.ndarray:
        '''
        역할: 첫번째 WEC의 Heave Stiffness(3, 3) 하나만 읽어서
            모든 WEC에 동일하게 적용한 대각 행렬을 반환
        '''
        fpath = self._get_path('wec.hst')

        target_i, target_j = 3, 3

        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                f.readline()

                for line in f:
                    parts = line.split()
                    if parts[0].isdigit() and parts[1].isdigit():
                        i, j = int(parts[0]), int(parts[1])

                        if i == target_i and j == target_j:
                            k_val = float(parts[2])
                            break
        
        k_dim = k_val * self.rho * self.g
        return np.eye(5) * k_dim
    

    def get_hydro_coeffs(self, target_indices: List[int]) -> Tuple[Dict, Dict, Dict]:
        '''
        역할: wec.1과 wec.2 파일을 벡터 연산으로 고속 파싱
        target_indices: 각 WEC의 Heave 모드 인덱스
        '''
        target_arr = np.array(target_indices)
        num_dof = len(target_indices)

        max_idx = target_arr.max()
        lookup = np.full(max_idx + 1, -1, dtype=int)
        lookup[target_arr] = np.arange(num_dof)

        added_mass, damping, exciting = {}, {}, {}

        fpath1 = self._get_path('wec.1')
        df1 = pd.read_csv(fpath1, sep=r'\s+', skiprows=1, header=None,
                          names=['freq', 'I', 'J', 'A', 'B'])
        df1['A'] *= self.rho
        df1['B'] *= (self.rho * df1['freq'])
        
        mask_range = (df1['I'] <= max_idx) & (df1['J'] <= max_idx)
        df1 = df1[mask_range]

        rows = lookup[df1['I'].values]
        cols = lookup[df1['J'].values]

        valid_mask = (rows != -1) & (cols != -1)

        ferqs = df1['freq'].values[valid_mask]
        r_idx = rows[valid_mask]
        c_idx = cols[valid_mask]
        A_vals = df1['A'].values[valid_mask]
        B_vals = df1['B'].values[valid_mask]

        unique_freqs = np.unique(ferqs)

        f_indices = np.searchsorted(unique_freqs, ferqs)

        n_freq = len(unique_freqs)
        A_3d = np.zeros((n_freq, num_dof, num_dof))
        B_3d = np.zeros((n_freq, num_dof, num_dof))

        A_3d[f_indices, r_idx, c_idx] = A_vals
        B_3d[f_indices, r_idx, c_idx] = B_vals
        
        for i, f in enumerate(unique_freqs):
            added_mass[f] = A_3d[i]
            damping[f] = B_3d[i]

        fpath2 = self._get_path('wec.2')
        df2 = pd.read_csv(fpath2, sep=r'\s+', skiprows=1, header=None,
                          names=['freq', 'Beta', 'I', 'Mag', 'Pha', 'Re', 'Im'])
        freqs_raw = df2['freq'].values
        I_raw = df2['I'].values
        Re_row = df2['Re'].values
        Im_row = df2['Im'].values

        complex_F = (Re_row + 1j * Im_row) * (self.rho * self.g)

        mask_range_2 =(I_raw <= max_idx)
        freqs_e = freqs_raw[mask_range_2]
        I_e = I_raw[mask_range_2]
        F_e = complex_F[mask_range_2]

        rows_e = lookup[I_e]
        valid_e = (rows_e != -1)

        final_freqs = freqs_e[valid_e]
        final_rows = rows_e[valid_e]
        final_F = F_e[valid_e]

        unique_freqs_e = np.unique(final_freqs)
        f_indices_e = np.searchsorted(unique_freqs_e, final_freqs)

        n_freq_e = len(unique_freqs_e)
        F_2d = np.zeros((n_freq_e, num_dof), dtype=complex)

        F_2d[f_indices_e, final_rows] = final_F

        for i, f in enumerate(unique_freqs_e):
            exciting[f] = F_2d[i]

        return added_mass, damping, exciting
    
    def calc_power(self, Hs, Tp, gamma) -> Tuple[float, List[float]]:
        mass = self.get_mass()
        stiffness = self.get_stiffness()
        target_indices = [3, 9, 15, 21, 27]
        added_mass, damping, exciting = self.get_hydro_coeffs(target_indices)
        M_matrix = np.eye(5) * mass
        freqs = sorted(added_mass.keys())

        ######################################################
        # PTO 없이 계산
        ######################################################
        rao_amplitudes = []

        for omega in freqs:
            A = added_mass[omega]
            B = damping[omega]
            F_exc = exciting[omega]

            term_inertia = -(omega**2) * (M_matrix + A)
            term_damping = 1j * omega * B
            
            Z_matrix = term_inertia + term_damping + stiffness

            displacement_complex = np.linalg.solve(Z_matrix, F_exc)
            rao_abs = np.abs(displacement_complex)

            row = [omega] + rao_abs.tolist()
            rao_amplitudes.append(row)
        
        columns = ['freq'] + [f'WEC_{i+1}' for i in range(5)]
        df_rao = pd.DataFrame(rao_amplitudes, columns=columns)

        ######################################################
        # PTO 적용 후 계산
        ######################################################
        peak_indices = df_rao.iloc[:, 1:].idxmax()
        
        pto_damping_values = []
        for i, col in enumerate(df_rao.columns[1:]):
            idx = peak_indices[col]
            wn = df_rao.loc[idx, 'freq']

            b_matrix_at_wn = damping[wn]
            b_val = b_matrix_at_wn[i, i]
            pto_damping_values.append(b_val)
        
        B_pto_matrix = np.diag(pto_damping_values)
        
        rao_amplitudes_pto = []
        for omega in freqs:
            A = added_mass[omega]
            B = damping[omega]
            F_exc = exciting[omega]

            term_inertia = -(omega**2) * (M_matrix + A)
            term_damping = 1j * omega * (B + B_pto_matrix)
            Z_matrix = term_inertia + term_damping + stiffness

            disp_complex = np.linalg.solve(Z_matrix, F_exc)
            disp_abs = np.abs(disp_complex)

            row = [omega] + disp_abs.tolist()
            rao_amplitudes_pto.append(row)

        df_rao_pto = pd.DataFrame(rao_amplitudes_pto, columns=columns)

        #######################################################
        # WEC별 발전량 계산
        #######################################################
        freq_array = df_rao_pto['freq'].values      # Shape: (N, )
        rao_matrix = df_rao_pto.iloc[:, 1:].values  # Shape: (N, 5)
        cpto_array = np.array(pto_damping_values)   # Shape: (5, )

        S = JONSWAP_SPEC(freq_array, Hs, Tp, gamma) # Shape: (N, )

        term_b = 0.5 * cpto_array                   # Shape: (5, )
        term_w = (freq_array[:, np.newaxis] ** 2)   # Shape: (N, 1)
        term_rao = rao_matrix ** 2                  # Shape: (N, 5)
        term_s = S[:, np.newaxis]                   # Shape: (N, 1)

        power_density = term_b * term_w * term_rao * term_s # Shape: (N, 5)
        
        individual_powers = simpson(power_density, freq_array, axis=0) # Shape: (5, )
        total_power = np.sum(individual_powers)
        
        return total_power, individual_powers
    
    def get_wavenumber(self, omega:np.ndarray, h:float)->np.ndarray:
        k_out = np.zeros_like(omega)
        k_guess = (omega**2) / self.g
        for i, w in enumerate(omega):
            if w <= 0:
                k_out[i] = 0
                continue
            func = lambda k: self.g * k * np.tanh(k * h) - w**2
            fprime = lambda k: self.g * np.tanh(k * h) + self.g * k * h * (1 / np.cosh(k * h))**2
            guess = k_guess[i] if k_guess[i] > 1e-6 else 1e-6
            k_out[i] = newton(func, guess, fprime=fprime)

        return k_out
    
    def group_vel(self, freqs:np.ndarray, h:float)->np.ndarray:
        k = self.get_wavenumber(freqs, h)
        with np.errstate(divide='ignore', invalid='ignore'):
            kh = k * h
            sinh_2kh = np.sinh(2 * kh)
            term_bracket = 1 + (2 * kh) / sinh_2kh
            cg = (freqs / (2 * k)) * term_bracket
        cg = np.nan_to_num(cg, nan=0.0)
        return cg
    
    def incident_power(self, Hs:float, Tp:float, gamma:float, h:float)->float:
        freq = np.linspace(0.5, 3.0, 51)  # rad/s
        S = JONSWAP_SPEC(freq, Hs, Tp, gamma)
        cg = self.group_vel(freq, h)

        integrand = self.rho * self.g * cg * S
        P_inc = simpson(integrand, freq)
        return P_inc