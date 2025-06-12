import heapq
import time
import random
from collections import deque
from typing import List, Tuple, Optional, Dict
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import os

class PuzzleState:
    """8パズルの状態を表現するクラス"""
    
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
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # 新しいボードを作成
                new_board = [row[:] for row in self.board]
                # タイルと空白を交換
                new_board[row][col] = new_board[new_row][new_col]
                new_board[new_row][new_col] = 0
                
                neighbor = PuzzleState(new_board, (new_row, new_col), 
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

class HeuristicFunctions:
    """ヒューリスティック関数群"""
    
    @staticmethod
    def h0(state: PuzzleState, goal_state: PuzzleState) -> int:
        """h0: 常に0を返す（課題2-8）"""
        return 0
    
    @staticmethod
    def h1(state: PuzzleState, goal_state: PuzzleState) -> int:
        """h1: ゴールの位置にないタイルの数"""
        misplaced = 0
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != 0:  # 空白以外
                    if state.board[i][j] != goal_state.board[i][j]:
                        misplaced += 1
        return misplaced
    
    @staticmethod
    def h2(state: PuzzleState, goal_state: PuzzleState) -> int:
        """h2: マンハッタン距離の和"""
        # ゴール状態での各タイルの位置を記録
        goal_positions = {}
        for i in range(3):
            for j in range(3):
                if goal_state.board[i][j] != 0:
                    goal_positions[goal_state.board[i][j]] = (i, j)
        
        manhattan_distance = 0
        for i in range(3):
            for j in range(3):
                tile = state.board[i][j]
                if tile != 0:  # 空白以外
                    goal_i, goal_j = goal_positions[tile]
                    manhattan_distance += abs(i - goal_i) + abs(j - goal_j)
        
        return manhattan_distance

class SearchAlgorithms:
    """探索アルゴリズム群"""
    
    def __init__(self):
        self.nodes_expanded = 0
        self.max_memory = 0
    
    def ids(self, initial_state: PuzzleState, goal_state: PuzzleState, max_depth: int = 30):
        """反復深化探索（IDS）"""
        self.nodes_expanded = 0
        self.max_memory = 0
        
        for depth_limit in range(max_depth + 1):
            result = self._depth_limited_search(initial_state, goal_state, depth_limit)
            if result is not None:
                return result
        return None
    
    def _depth_limited_search(self, initial_state: PuzzleState, goal_state: PuzzleState, limit: int):
        """深さ制限探索"""
        stack = [initial_state]
        visited = set()
        
        while stack:
            self.max_memory = max(self.max_memory, len(stack) + len(visited))
            current = stack.pop()
            
            if current.is_goal(goal_state):
                return current
            
            if current.cost < limit:
                visited.add(current.board_tuple)
                self.nodes_expanded += 1
                
                for neighbor in current.get_neighbors():
                    if neighbor.board_tuple not in visited:
                        stack.append(neighbor)
        
        return None
    
    def a_star(self, initial_state: PuzzleState, goal_state: PuzzleState, heuristic_func):
        """A*探索"""
        self.nodes_expanded = 0
        self.max_memory = 0
        
        # f(n) = g(n) + h(n)でソートされる優先度付きキュー
        open_list = [(heuristic_func(initial_state, goal_state), 0, initial_state)]
        closed_set = set()
        g_costs = {initial_state.board_tuple: 0}
        
        while open_list:
            self.max_memory = max(self.max_memory, len(open_list) + len(closed_set))
            _, current_g, current = heapq.heappop(open_list)
            
            if current.is_goal(goal_state):
                return current
            
            if current.board_tuple in closed_set:
                continue
                
            closed_set.add(current.board_tuple)
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

def create_random_puzzle(goal_state: PuzzleState, steps: int = 100) -> PuzzleState:
    """ゴール状態からランダムに動かして初期状態を作成"""
    current = PuzzleState([row[:] for row in goal_state.board], goal_state.empty_pos)
    
    for _ in range(steps):
        neighbors = current.get_neighbors()
        current = random.choice(neighbors)
        current.parent = None  # 親の参照をクリア
        current.cost = 0
    
    return current

def run_experiments(num_trials: int = 50):
    """実験を実行し、結果を収集"""
    # ゴール状態
    goal_board = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
    goal_state = PuzzleState(goal_board, (1, 1))
    
    # ヒューリスティック関数
    heuristics = {
        'h0': HeuristicFunctions.h0,
        'h1': HeuristicFunctions.h1,
        'h2': HeuristicFunctions.h2
    }
    
    results = {
        'IDS': {'nodes': [], 'time': [], 'memory': [], 'solution_length': []},
        'A*_h0': {'nodes': [], 'time': [], 'memory': [], 'solution_length': []},
        'A*_h1': {'nodes': [], 'time': [], 'memory': [], 'solution_length': []},
        'A*_h2': {'nodes': [], 'time': [], 'memory': [], 'solution_length': []}
    }
    
    search_algo = SearchAlgorithms()
    
    print(f"実験開始: {num_trials}回のトライアル")
    
    for trial in range(num_trials):
        if trial % 10 == 0:
            print(f"トライアル {trial}/{num_trials}")
        
        # ランダムな初期状態を生成
        initial_state = create_random_puzzle(goal_state, random.randint(20, 100))
        
        # IDS
        start_time = time.time()
        solution = search_algo.ids(initial_state, goal_state)
        end_time = time.time()
        
        if solution:
            results['IDS']['nodes'].append(search_algo.nodes_expanded)
            results['IDS']['time'].append(end_time - start_time)
            results['IDS']['memory'].append(search_algo.max_memory)
            results['IDS']['solution_length'].append(len(solution.get_path()))
        
        # A* with different heuristics
        for h_name, h_func in heuristics.items():
            start_time = time.time()
            solution = search_algo.a_star(initial_state, goal_state, h_func)
            end_time = time.time()
            
            if solution:
                key = f'A*_{h_name}'
                results[key]['nodes'].append(search_algo.nodes_expanded)
                results[key]['time'].append(end_time - start_time)
                results[key]['memory'].append(search_algo.max_memory)
                results[key]['solution_length'].append(len(solution.get_path()))
    
    return results

def analyze_results(results: Dict):
    """結果を分析し、統計情報を計算"""
    analysis = {}
    
    for algorithm, metrics in results.items():
        analysis[algorithm] = {}
        for metric, values in metrics.items():
            if values:  # 空でない場合
                analysis[algorithm][metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return analysis

def create_report(results: Dict, analysis: Dict):
    """レポートを生成"""
    print("\n" + "="*80)
    print("8パズル探索アルゴリズム性能比較レポート")
    print("="*80)
    
    # 成功率の計算
    total_trials = 1000  # 想定される試行回数
    print("\n1. 成功率")
    print("-" * 40)
    for algorithm in results.keys():
        success_rate = len(results[algorithm]['nodes']) / total_trials * 100
        print(f"{algorithm}: {success_rate:.1f}% ({len(results[algorithm]['nodes'])}/{total_trials})")
    
    # 平均性能の比較
    print("\n2. 平均性能比較")
    print("-" * 40)
    
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
                print(f"  {algorithm}: {mean_val:.2f} ± {stdev_val:.2f}")
    
    # 効率性の分析
    print("\n3. 効率性分析")
    print("-" * 40)
    
    # ヒューリスティック関数の比較
    if 'A*_h1' in analysis and 'A*_h2' in analysis:
        h1_nodes = analysis['A*_h1']['nodes']['mean'] if 'nodes' in analysis['A*_h1'] else float('inf')
        h2_nodes = analysis['A*_h2']['nodes']['mean'] if 'nodes' in analysis['A*_h2'] else float('inf')
        
        print(f"ヒューリスティック関数の比較:")
        print(f"  h1 (misplaced tiles): 平均{h1_nodes:.1f}ノード展開")
        print(f"  h2 (Manhattan distance): 平均{h2_nodes:.1f}ノード展開")
        
        if h2_nodes < h1_nodes:
            improvement = (h1_nodes - h2_nodes) / h1_nodes * 100
            print(f"  h2はh1より{improvement:.1f}%効率的")
        elif h1_nodes < h2_nodes:
            improvement = (h2_nodes - h1_nodes) / h2_nodes * 100
            print(f"  h1はh2より{improvement:.1f}%効率的")
    
    return analysis

def create_visualizations(results: Dict, analysis: Dict):
    """結果の可視化"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 日本語フォント設定（環境に応じて調整）
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # グラフのサイズと余白を調整
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('8-Puzzle Search Algorithm Performance Comparison', fontsize=16, y=0.95)
    
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
            ax.set_title(metric_name, pad=20)  # タイトルとグラフの間のスペースを増やす
            ax.set_ylabel(metric_name, labelpad=10)  # y軸ラベルとの間隔を増やす
            
            # バーの上に値を表示
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.2f}', ha='center', va='bottom')
            
            # ログスケールを適用（ノード数と時間の場合）
            if metric in ['nodes', 'time']:
                ax.set_yscale('log')
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)  # グリッド線を追加
    
    # 余白を調整
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 上部のタイトル用に余白を確保
    
    # グラフを保存
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'performance_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nグラフを保存しました: {output_path}")
    
    plt.show()

# メイン実行部分
if __name__ == "__main__":
    print("8パズル探索アルゴリズム性能比較実験")
    print("実装アルゴリズム: IDS, A*(h0), A*(h1), A*(h2)")
    
    # 実験実行
    results = run_experiments(num_trials=1000)
    
    # 結果分析
    analysis = analyze_results(results)
    
    # レポート生成
    report = create_report(results, analysis)
    
    # 可視化
    create_visualizations(results, analysis)
    
    print("\n実験完了！")