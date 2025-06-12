import heapq
import time
import random
from collections import deque
from typing import List, Tuple, Optional, Dict
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class Puzzle15State:
    """15パズルの状態を表現するクラス"""
    
    def __init__(self, board: List[List[int]], empty_pos: Tuple[int, int], 
                 parent=None, move: str = "", cost: int = 0):
        self.board = board
        self.empty_pos = empty_pos  # 空白の位置 (row, col)
        self.parent = parent
        self.move = move
        self.cost = cost
        self.board_tuple = tuple(tuple(row) for row in board)
    
    def __eq__(self, other):
        return self.board_tuple == other.board_tuple
    
    def __hash__(self):
        return hash(self.board_tuple)
    
    def __lt__(self, other):
        return False  # heapqのために必要
    
    def is_goal(self, goal_state):
        return self.board_tuple == goal_state.board_tuple
    
    def get_neighbors(self):
        """隣接状態を生成"""
        neighbors = []
        row, col = self.empty_pos
        moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
        
        for dr, dc, move_name in moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 4 and 0 <= new_col < 4:
                # 新しいボードを作成
                new_board = [row[:] for row in self.board]
                # タイルと空白を交換
                new_board[row][col] = new_board[new_row][new_col]
                new_board[new_row][new_col] = 0
                
                neighbor = Puzzle15State(new_board, (new_row, new_col), 
                                       self, move_name, self.cost + 1)
                neighbors.append(neighbor)
        
        return neighbors
    
    def get_path(self):
        """解への経路を取得"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.move)
            current = current.parent
        return list(reversed(path))

class HeuristicFunctions15:
    """15パズル用ヒューリスティック関数群"""
    
    @staticmethod
    def h0(state: Puzzle15State, goal_state: Puzzle15State) -> int:
        """h0: 常に0を返す"""
        return 0
    
    @staticmethod
    def h1(state: Puzzle15State, goal_state: Puzzle15State) -> int:
        """h1: ゴールの位置にないタイルの数"""
        misplaced = 0
        for i in range(4):
            for j in range(4):
                if state.board[i][j] != 0:  # 空白以外
                    if state.board[i][j] != goal_state.board[i][j]:
                        misplaced += 1
        return misplaced
    
    @staticmethod
    def h2(state: Puzzle15State, goal_state: Puzzle15State) -> int:
        """h2: マンハッタン距離の和"""
        # ゴール状態での各タイルの位置を記録
        goal_positions = {}
        for i in range(4):
            for j in range(4):
                if goal_state.board[i][j] != 0:
                    goal_positions[goal_state.board[i][j]] = (i, j)
        
        manhattan_distance = 0
        for i in range(4):
            for j in range(4):
                tile = state.board[i][j]
                if tile != 0:  # 空白以外
                    goal_i, goal_j = goal_positions[tile]
                    manhattan_distance += abs(i - goal_i) + abs(j - goal_j)
        
        return manhattan_distance
    
    @staticmethod
    def h3(state: Puzzle15State, goal_state: Puzzle15State) -> int:
        """h3: マンハッタン距離 + 線形競合（Linear Conflict）"""
        manhattan = HeuristicFunctions15.h2(state, goal_state)

        goal_positions = {}
        for r in range(4):
            for c in range(4):
                if goal_state.board[r][c] != 0:
                    goal_positions[goal_state.board[r][c]] = (r, c)

        linear_conflict = 0

        # 行の線形競合
        for row_idx in range(4):
            # この行にあるタイルで、かつ目標行もこの行であるタイルのリスト
            tiles_in_correct_row = []
            for col_idx in range(4):
                tile = state.board[row_idx][col_idx]
                if tile != 0:
                    goal_r, goal_c = goal_positions[tile]
                    if goal_r == row_idx:
                        tiles_in_correct_row.append((tile, col_idx)) # (タイル値, 現在の列)

            # 線形競合のチェック
            for i in range(len(tiles_in_correct_row)):
                for j in range(i + 1, len(tiles_in_correct_row)):
                    tile1_val, tile1_curr_c = tiles_in_correct_row[i]
                    tile2_val, tile2_curr_c = tiles_in_correct_row[j]

                    # 同じ行にあり、目標の行も同じだが、目標の列順が逆になっている場合
                    if goal_positions[tile1_val][1] > goal_positions[tile2_val][1]:
                        linear_conflict += 2

        # 列の線形競合 (同様のロジック)
        for col_idx in range(4):
            tiles_in_correct_col = []
            for row_idx in range(4):
                tile = state.board[row_idx][col_idx]
                if tile != 0:
                    goal_r, goal_c = goal_positions[tile]
                    if goal_c == col_idx:
                        tiles_in_correct_col.append((tile, row_idx)) # (タイル値, 現在の行)

            for i in range(len(tiles_in_correct_col)):
                for j in range(i + 1, len(tiles_in_correct_col)):
                    tile1_val, tile1_curr_r = tiles_in_correct_col[i]
                    tile2_val, tile2_curr_r = tiles_in_correct_col[j]

                    # 同じ列にあり、目標の列も同じだが、目標の行順が逆になっている場合
                    if goal_positions[tile1_val][0] > goal_positions[tile2_val][0]:
                        linear_conflict += 2

        return manhattan + linear_conflict
    
    # @staticmethod
    # def h3(state: Puzzle15State, goal_state: Puzzle15State) -> int:
    #     """h3: マンハッタン距離 + 線形競合（Linear Conflict）"""
    #     # 基本のマンハッタン距離を計算
    #     manhattan = HeuristicFunctions15.h2(state, goal_state)
        
    #     # ゴール状態での各タイルの位置を記録
    #     goal_positions = {}
    #     for i in range(4):
    #         for j in range(4):
    #             if goal_state.board[i][j] != 0:
    #                 goal_positions[goal_state.board[i][j]] = (i, j)
        
    #     linear_conflict = 0
        
    #     # 行の線形競合をチェック
    #     for row in range(4):
    #         max_val = -1
    #         for col in range(4):
    #             tile = state.board[row][col]
    #             if tile != 0 and tile in goal_positions:
    #                 goal_row, goal_col = goal_positions[tile]
    #                 if goal_row == row:  # 正しい行にある
    #                     if goal_col < max_val:
    #                         linear_conflict += 2  # 2移動が必要
    #                     else:
    #                         max_val = goal_col
        
    #     # 列の線形競合をチェック
    #     for col in range(4):
    #         max_val = -1
    #         for row in range(4):
    #             tile = state.board[row][col]
    #             if tile != 0 and tile in goal_positions:
    #                 goal_row, goal_col = goal_positions[tile]
    #                 if goal_col == col:  # 正しい列にある
    #                     if goal_row < max_val:
    #                         linear_conflict += 2
    #                     else:
    #                         max_val = goal_row
        
    #     return manhattan + linear_conflict
    
    @staticmethod
    def h4(state: Puzzle15State, goal_state: Puzzle15State) -> int:
        """h4: 行/列不一致数ヒューリスティック (Walking Distanceの近似)"""
        # 各行・列で目標位置の行/列と異なるタイル数をカウント
        row_mismatches = [0] * 4
        col_mismatches = [0] * 4
        
        goal_positions = {}
        for i in range(4):
            for j in range(4):
                if goal_state.board[i][j] != 0:
                    goal_positions[goal_state.board[i][j]] = (i, j)
        
        for i in range(4):
            for j in range(4):
                tile = state.board[i][j]
                if tile != 0 and tile in goal_positions:
                    goal_i, goal_j = goal_positions[tile]
                    if goal_i != i: # タイルが目標の行にない場合
                        row_mismatches[i] += 1
                    if goal_j != j: # タイルが目標の列にない場合
                        col_mismatches[j] += 1
        
        # 各行・列の不一致数を合計
        total_mismatches = sum(row_mismatches) + sum(col_mismatches)
        
        # マンハッタン距離との最大値を取る（より強力なヒューリスティックに）
        manhattan = HeuristicFunctions15.h2(state, goal_state)
        return max(manhattan, total_mismatches)
    # @staticmethod
    # def h4(state: Puzzle15State, goal_state: Puzzle15State) -> int:
    #     """h4: Walking Distance（歩行距離）の近似"""
    #     # 各行・列で正しい位置にないタイルの数をカウント
    #     row_conflicts = [0] * 4
    #     col_conflicts = [0] * 4
        
    #     goal_positions = {}
    #     for i in range(4):
    #         for j in range(4):
    #             if goal_state.board[i][j] != 0:
    #                 goal_positions[goal_state.board[i][j]] = (i, j)
        
    #     for i in range(4):
    #         for j in range(4):
    #             tile = state.board[i][j]
    #             if tile != 0 and tile in goal_positions:
    #                 goal_i, goal_j = goal_positions[tile]
    #                 if goal_i != i:
    #                     row_conflicts[i] += 1
    #                 if goal_j != j:
    #                     col_conflicts[j] += 1
        
    #     # Walking Distanceの近似として、各行・列の競合数の合計を使用
    #     walking_distance = sum(row_conflicts) + sum(col_conflicts)
        
    #     # マンハッタン距離との最大値を取る
    #     manhattan = HeuristicFunctions15.h2(state, goal_state)
    #     return max(manhattan, walking_distance)

class SearchAlgorithms15:
    """15パズル用探索アルゴリズム群"""
    
    def __init__(self):
        self.nodes_expanded = 0
        self.max_memory = 0
        self.time_limit = 60.0  # 時間制限（秒）
    
    def ids(self, initial_state: Puzzle15State, goal_state: Puzzle15State, max_depth: int = 25):
        """反復深化探索（IDS） - 時間制限付き"""
        self.nodes_expanded = 0
        self.max_memory = 0
        start_time = time.time()
        
        for depth_limit in range(max_depth + 1):
            if time.time() - start_time > self.time_limit:
                return None  # タイムアウト
            
            result = self._depth_limited_search(initial_state, goal_state, depth_limit, start_time)
            if result is not None:
                return result
        return None
    
    def _depth_limited_search(self, initial_state: Puzzle15State, goal_state: Puzzle15State, 
                             limit: int, start_time: float):
        """深さ制限探索"""
        stack = [initial_state]
        visited = set()
        
        while stack:
            if time.time() - start_time > self.time_limit:
                return None  # タイムアウト
                
            self.max_memory = max(self.max_memory, len(stack) + len(visited))
            current = stack.pop()
            
            if current.is_goal(goal_state):
                return current
            
            if current.cost < limit:
                state_key = current.board_tuple
                if state_key not in visited:
                    visited.add(state_key)
                    self.nodes_expanded += 1
                    
                    for neighbor in current.get_neighbors():
                        if neighbor.board_tuple not in visited:
                            stack.append(neighbor)
        
        return None
    
    def a_star(self, initial_state: Puzzle15State, goal_state: Puzzle15State, heuristic_func):
        """A*探索 - 時間制限付き"""
        self.nodes_expanded = 0
        self.max_memory = 0
        start_time = time.time()
        
        # f(n) = g(n) + h(n)でソートされる優先度付きキュー
        open_list = [(heuristic_func(initial_state, goal_state), 0, initial_state)]
        closed_set = set()
        g_costs = {initial_state.board_tuple: 0}
        
        while open_list:
            if time.time() - start_time > self.time_limit:
                return None  # タイムアウト
                
            self.max_memory = max(self.max_memory, len(open_list) + len(closed_set))
            _, current_g, current = heapq.heappop(open_list)
            
            if current.is_goal(goal_state):
                return current
            
            current_tuple = current.board_tuple
            if current_tuple in closed_set:
                continue
                
            closed_set.add(current_tuple)
            self.nodes_expanded += 1
            
            for neighbor in current.get_neighbors():
                neighbor_tuple = neighbor.board_tuple
                
                if neighbor_tuple in closed_set:
                    continue
                
                tentative_g = current_g + 1
                
                if neighbor_tuple not in g_costs or tentative_g < g_costs[neighbor_tuple]:
                    g_costs[neighbor_tuple] = tentative_g
                    neighbor.cost = tentative_g
                    f_cost = tentative_g + heuristic_func(neighbor, goal_state)
                    heapq.heappush(open_list, (f_cost, tentative_g, neighbor))
        
        return None

def create_random_puzzle15(goal_state: Puzzle15State, steps: int = 200) -> Puzzle15State:
    """ゴール状態からランダムに動かして初期状態を作成"""
    current = Puzzle15State([row[:] for row in goal_state.board], goal_state.empty_pos)
    
    for _ in range(steps):
        neighbors = current.get_neighbors()
        current = random.choice(neighbors)
        current.parent = None  # 親の参照をクリア
        current.cost = 0
    
    return current

def is_solvable(puzzle_state: Puzzle15State) -> bool:
    """15パズルが解けるかどうかを判定"""
    # ボードを1次元リストに変換（0を除く）
    flat_board = []
    empty_row = 0
    
    for i in range(4):
        for j in range(4):
            if puzzle_state.board[i][j] == 0:
                empty_row = i
            else:
                flat_board.append(puzzle_state.board[i][j])
    
    # 反転数を計算
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1
    
    # 15パズルの解けるかどうかの判定
    # 空白が奇数行（下から数えて）にあり、反転数が偶数の場合、または
    # 空白が偶数行にあり、反転数が奇数の場合に解ける
    blank_row_from_bottom = 4 - empty_row
    
    if blank_row_from_bottom % 2 == 1:  # 奇数行
        return inversions % 2 == 0
    else:  # 偶数行
        return inversions % 2 == 1

def run_experiments_15(num_trials: int = 20):
    """15パズルの実験を実行"""
    # ゴール状態
    goal_board = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]
    goal_state = Puzzle15State(goal_board, (3, 3))
    
    # ヒューリスティック関数
    heuristics = {
        'h0': HeuristicFunctions15.h0,
        'h1': HeuristicFunctions15.h1,
        'h2': HeuristicFunctions15.h2,
        'h3': HeuristicFunctions15.h3,
        'h4': HeuristicFunctions15.h4
    }
    
    results = {
        'IDS': {'nodes': [], 'time': [], 'memory': [], 'solution_length': [], 'timeouts': 0},
        'A*_h0': {'nodes': [], 'time': [], 'memory': [], 'solution_length': [], 'timeouts': 0},
        'A*_h1': {'nodes': [], 'time': [], 'memory': [], 'solution_length': [], 'timeouts': 0},
        'A*_h2': {'nodes': [], 'time': [], 'memory': [], 'solution_length': [], 'timeouts': 0},
        'A*_h3': {'nodes': [], 'time': [], 'memory': [], 'solution_length': [], 'timeouts': 0},
        'A*_h4': {'nodes': [], 'time': [], 'memory': [], 'solution_length': [], 'timeouts': 0}
    }
    
    search_algo = SearchAlgorithms15()
    
    print(f"15パズル実験開始: {num_trials}回のトライアル")
    print("各アルゴリズムには60秒の時間制限があります")
    
    successful_puzzles = []
    
    for trial in range(num_trials):
        print(f"\nトライアル {trial + 1}/{num_trials}")
        
        # 解けるパズルが見つかるまで生成を続ける
        max_attempts = 50
        for attempt in range(max_attempts):
            initial_state = create_random_puzzle15(goal_state, random.randint(50, 200))
            if is_solvable(initial_state):
                break
        else:
            print("解けるパズルの生成に失敗しました")
            continue
        
        successful_puzzles.append(initial_state)
        
        print("初期状態:")
        for row in initial_state.board:
            print([f"{x:2d}" if x != 0 else "  " for x in row])
        
        # IDS（時間制限があるため、最初にテスト）
        print("IDS実行中...")
        start_time = time.time()
        solution = search_algo.ids(initial_state, goal_state)
        end_time = time.time()
        
        if solution:
            results['IDS']['nodes'].append(search_algo.nodes_expanded)
            results['IDS']['time'].append(end_time - start_time)
            results['IDS']['memory'].append(search_algo.max_memory)
            results['IDS']['solution_length'].append(len(solution.get_path()))
            print(f"IDS: {search_algo.nodes_expanded}ノード, {end_time - start_time:.2f}秒, 解長{len(solution.get_path())}")
        else:
            results['IDS']['timeouts'] += 1
            print("IDS: タイムアウト")
        
        # A* with different heuristics
        for h_name, h_func in heuristics.items():
            print(f"A*({h_name})実行中...")
            start_time = time.time()
            solution = search_algo.a_star(initial_state, goal_state, h_func)
            end_time = time.time()
            
            key = f'A*_{h_name}'
            if solution:
                results[key]['nodes'].append(search_algo.nodes_expanded)
                results[key]['time'].append(end_time - start_time)
                results[key]['memory'].append(search_algo.max_memory)
                results[key]['solution_length'].append(len(solution.get_path()))
                print(f"A*({h_name}): {search_algo.nodes_expanded}ノード, {end_time - start_time:.2f}秒, 解長{len(solution.get_path())}")
            else:
                results[key]['timeouts'] += 1
                print(f"A*({h_name}): タイムアウト")
    
    return results, successful_puzzles

def analyze_results_15(results: Dict):
    """15パズル結果の分析"""
    analysis = {}
    
    for algorithm, metrics in results.items():
        analysis[algorithm] = {}
        for metric, values in metrics.items():
            if metric != 'timeouts' and values:  # タイムアウト数以外で空でない場合
                analysis[algorithm][metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        # タイムアウト情報を追加
        analysis[algorithm]['timeouts'] = metrics['timeouts']
    
    return analysis

def create_report_15(results: Dict, analysis: Dict, num_trials: int):
    """15パズルレポートを生成"""
    print("\n" + "="*80)
    print("15パズル探索アルゴリズム性能比較レポート")
    print("="*80)
    
    # 成功率の計算
    print("\n1. 成功率（60秒制限内での解発見率）")
    print("-" * 50)
    for algorithm in results.keys():
        success_count = len(results[algorithm]['nodes'])
        timeout_count = results[algorithm]['timeouts']
        success_rate = success_count / num_trials * 100
        print(f"{algorithm}: {success_rate:.1f}% ({success_count}/{num_trials}) [タイムアウト: {timeout_count}]")
    
    # 平均性能の比較
    print("\n2. 平均性能比較（成功したケースのみ）")
    print("-" * 50)
    
    metrics_names = {
        'nodes': '展開ノード数',
        'time': '実行時間(秒)',
        'memory': 'メモリ使用量',
        'solution_length': '解の長さ'
    }
    
    for metric, metric_name in metrics_names.items():
        print(f"\n{metric_name}:")
        for algorithm in analysis.keys():
            if metric in analysis[algorithm] and analysis[algorithm][metric]['count'] > 0:
                mean_val = analysis[algorithm][metric]['mean']
                stdev_val = analysis[algorithm][metric]['stdev']
                count = analysis[algorithm][metric]['count']
                print(f"  {algorithm}: {mean_val:.2f} ± {stdev_val:.2f} (n={count})")
    
    # 効率性の分析
    print("\n3. 効率性分析")
    print("-" * 50)
    
    # 成功したケースでのヒューリスティック関数の比較
    h_algorithms = ['A*_h1', 'A*_h2', 'A*_h3', 'A*_h4']
    successful_h = []
    
    for alg in h_algorithms:
        if alg in analysis and 'nodes' in analysis[alg] and analysis[alg]['nodes']['count'] > 0:
            successful_h.append((alg, analysis[alg]['nodes']['mean']))
    
    if len(successful_h) > 1:
        successful_h.sort(key=lambda x: x[1])  # ノード数でソート
        print("ヒューリスティック関数の効率性ランキング（展開ノード数順）:")
        for i, (alg, nodes) in enumerate(successful_h):
            print(f"  {i+1}位: {alg} - 平均{nodes:.0f}ノード")
    
    return analysis

def create_visualizations_15(results: Dict, analysis: Dict):
    """15パズルの結果を可視化"""
    # 日本語フォント設定（環境に応じて調整）
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # グラフのサイズと余白を調整
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('15-Puzzle Search Algorithm Performance Comparison', fontsize=16, y=0.95)
    
    # データの準備
    algorithms = list(analysis.keys())
    metrics = ['nodes', 'time', 'memory', 'solution_length']
    metric_names = ['Nodes Expanded', 'Execution Time (s)', 'Memory Usage', 'Solution Length']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        means = []
        stdevs = []
        labels = []
        
        for alg in algorithms:
            if metric in analysis[alg] and analysis[alg][metric]['count'] > 0:
                means.append(analysis[alg][metric]['mean'])
                stdevs.append(analysis[alg][metric]['stdev'])
                labels.append(alg)
        
        if means:
            bars = ax.bar(labels, means, yerr=stdevs, capsize=5, alpha=0.7)
            ax.set_title(metric_name, pad=20)
            ax.set_ylabel(metric_name, labelpad=10)
            
            # バーの上に値を表示
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.2f}', ha='center', va='bottom')
            
            # ログスケールを適用（ノード数と時間の場合）
            if metric in ['nodes', 'time']:
                ax.set_yscale('log')
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 余白を調整
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # グラフを保存
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'15puzzle_performance_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nグラフを保存しました: {output_path}")
    
    plt.show()

# メイン実行
if __name__ == "__main__":
    print("15パズル探索アルゴリズム性能比較実験")
    print("実装アルゴリズム: IDS, A*(h0), A*(h1), A*(h2), A*(h3), A*(h4)")
    print("h3: Linear Conflict, h4: Walking Distance近似")
    
    # 実験実行（時間がかかるため試行回数を削減）
    results, puzzles = run_experiments_15(num_trials=100)
    
    # 結果分析
    analysis = analyze_results_15(results)
    
    # レポート生成
    report = create_report_15(results, analysis, num_trials=100)
    
    # 可視化
    create_visualizations_15(results, analysis)
    
    print("\n実験完了！")