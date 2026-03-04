import numpy as np
import shutil
from typing import List

# =============================================================================
# 📝 파일 기록 및 결과 관리 함수
# =============================================================================
def write_results(result_vector:List, results_path:str):
    """
    역할: 계산된 결과 데이터 한 줄을 지정된 파일 끝에 추가로 기록합니다.
    Input:
        - result_vector: 기록할 데이터 리스트 (좌표, 성능, 파워 등)
        - results_path: 저장할 파일 경로 (.res 또는 .csv)
    Output: 없음 (파일 쓰기 수행)
    """
    with open(results_path, 'a') as f:
        f.write(', '.join(map(str, result_vector)) + '\n')

def move_results_file(algorithm:str, loc_name:str):
    """
    역할: 최적화 완료 후, 임시 결과 파일들을 지역명과 알고리즘명이 포함된 고유 이름으로 변경합니다.
    Input:
        - loc_name: 실험 지역 명칭 (예: 'Incheon')
        - algorithm: 사용된 알고리즘 명칭 (예: 'DEPSO')
    Output: 없음 (파일 이동 및 이름 변경 수행)
    """
    shutil.move('geo_layout_cal_results.res', f'{algorithm}_{loc_name}_cal.res')
    shutil.move('geo_layout_iter_results.res', f'{algorithm}_{loc_name}_iter.res') 

# =============================================================================
# 📢 터미널 로그 출력 (Logging) 함수
# =============================================================================
def print_start_message(idx:int, vector:np.ndarray):
    """
    역할: 최적화 시작 단계에서 각 개체의 초기 위치 정보를 화면에 출력합니다.
    Input: idx (개체 번호), vector (좌표 배열)
    """
    print("-" * 60 + f'\n 💡 Initial Position {idx + 1}: {np.round(vector, 1)}')

def print_eval_message(elapsed):
    """
    역할: 한 번의 성능 평가(목적함수 계산)가 완료되었을 때 소요 시간을 출력합니다.
    Input: elapsed (소요 시간, 초 단위)
    """
    print(f"      ⏱️ Evaluation finished in {elapsed:.2f} seconds.")

def print_iter_start_message(func_name:str, location:str, Hs:float, Tp:float, h:float):
    """
    역할: 최적화 실험의 기본 정보와 시작을 알리는 헤더를 출력합니다.
    Input: func_name (알고리즘명), location (지역), Hs (파고), Tp (주기), h (수심)
    """
    print("=" * 70)
    print(f"🚀 {func_name} Optimization Start")
    print(f"📍 Location: {location} | Hs: {Hs} m, Tp: {Tp} s, h: {h} m")
    print("=" * 70)

def print_summary_message(iteration: int, gbest: np.ndarray, gbest_fitness: float, total_time: float, cnt_memory: int):
    """
    역할: 매 반복(Iteration) 단계 종료 시 현재까지의 최적 성과와 통계 정보를 요약 출력합니다.
    Input:
        - iteration: 현재 반복 횟수
        - gbest: 현재까지의 전역 최적 위치
        - gbest_fitness: 현재까지의 최고 성능값
        - total_time: 반복에 소요된 총 시간 (분)
        - cnt_memory: 메모리 참조(중복 계산 방지) 횟수
    """
    print(f'\n--- Iteration {iteration} Summary ---')
    print(f'  🌟 Best Position: {np.round(gbest, 1)}')
    print(f'  🏆 Best Fitness : {gbest_fitness:.3f}')
    print(f'  ⏳ Elapsed Time  : {total_time:.2f} min')
    print(f'  🧠 Memory Hits  : {cnt_memory}')

def distance_check(vector:np.ndarray) -> bool:
    x = vector[[0, 2, 4, 2, 0]]
    y = vector[[1, 3, 5, 3, 1]]
    y[3:] *= -1

    points = np.stack((x, y)).T
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)

    np.fill_diagonal(dist_sq, np.inf)

    min_dist_sq = np.min(dist_sq)
    
    min_distance = 4 * vector[6] # WEC1, WEC2 반지름 + D 만큼 반격
    
    return min_dist_sq > (min_distance ** 2)

def apply_step_size(vector:np.ndarray, step_size:np.ndarray) -> np.ndarray:
    modified_vector = np.round(vector / step_size) * step_size
    modified_vector = np.round(modified_vector, 1)
    return modified_vector

def normalize_layout_vector(vector:np.ndarray) -> np.ndarray:
    if (vector[1], vector[0]) < (vector[3], vector[2]):
        vector[[0, 2]] = vector[[2, 0]]
        vector[[1, 3]] = vector[[3, 1]]
    return vector

if __name__ == "__main__":
    vector = np.array([3, 30, 3, 15, 3, 0, 2.5, 4.8])
    distance_check(vector)