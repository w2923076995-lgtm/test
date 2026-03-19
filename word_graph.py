import argparse
import os
import re
import random
import heapq
from collections import Counter, defaultdict, deque


class WordGraph:
    def __init__(self, words):
        self.words = words
        self.word_freq = Counter(words)              # 单词出现次数
        self.graph = defaultdict(Counter)           # graph[a][b] = a -> b 的权重
        self.reverse_graph = defaultdict(Counter)   # reverse_graph[b][a] = a -> b 的权重

        for a, b in zip(words, words[1:]):
            self.graph[a][b] += 1
            self.reverse_graph[b][a] += 1

    @property
    def node_count(self):
        return len(self.word_freq)

    @property
    def edge_count(self):
        return sum(len(neighbors) for neighbors in self.graph.values())

    @property
    def total_transitions(self):
        return max(0, len(self.words) - 1)

    def print_summary(self):
        print("\n========== 图摘要 ==========")
        print(f"总单词数（含重复）: {len(self.words)}")
        print(f"不同单词数（节点数）: {self.node_count}")
        print(f"有向边数: {self.edge_count}")
        print(f"相邻关系总次数: {self.total_transitions}")
        print("===========================\n")

    def print_graph(self):
        """
        功能需求2：在 CLI 上清晰展示有向图
        自定义格式：每个节点单独成块，下面列出它的所有出边及权重
        """
        print("\n==================== 有向图（CLI 展示） ====================")

        all_nodes = sorted(self.word_freq.keys())

        for i, node in enumerate(all_nodes, start=1):
            print(f"[{i}] 节点: {node}")
            print(f"    出现次数: {self.word_freq[node]}")

            out_neighbors = self.graph.get(node, {})
            if out_neighbors:
                print("    出边:")
                sorted_neighbors = sorted(out_neighbors.items(), key=lambda x: (-x[1], x[0]))
                for j, (nbr, wt) in enumerate(sorted_neighbors, start=1):
                    print(f"      ({j}) {node} -> {nbr}    weight = {wt}")
            else:
                print("    出边: 无")

            print("------------------------------------------------------------")

        print("================== CLI 图结构展示结束 ==================\n")

    def print_graph_compact(self):
        """
        紧凑版邻接表输出
        """
        print("\n========== 图结构（紧凑邻接表） ==========")
        all_nodes = sorted(self.word_freq.keys())
        for node in all_nodes:
            if node in self.graph and self.graph[node]:
                neighbors = sorted(self.graph[node].items(), key=lambda x: (-x[1], x[0]))
                edge_str = ", ".join([f"{nbr}(w={wt})" for nbr, wt in neighbors])
                print(f"{node:<15} -> {edge_str}")
            else:
                print(f"{node:<15} -> [无出边]")
        print("=========================================\n")

    def query_word(self, word):
        word = word.lower()

        print(f"\n========== 单词查询: {word} ==========")
        if word not in self.word_freq:
            print("该单词不在图中。")
            print("====================================\n")
            return

        print(f"出现次数: {self.word_freq[word]}")

        out_neighbors = self.graph.get(word, {})
        in_neighbors = self.reverse_graph.get(word, {})

        weighted_out_degree = sum(out_neighbors.values())
        weighted_in_degree = sum(in_neighbors.values())

        print(f"不同后继节点数（出度）: {len(out_neighbors)}")
        print(f"不同前驱节点数（入度）: {len(in_neighbors)}")
        print(f"加权出度: {weighted_out_degree}")
        print(f"加权入度: {weighted_in_degree}")

        print("\n后继节点：")
        if out_neighbors:
            for nbr, wt in sorted(out_neighbors.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {word} -> {nbr} (w={wt})")
        else:
            print("  无")

        print("\n前驱节点：")
        if in_neighbors:
            for nbr, wt in sorted(in_neighbors.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {nbr} -> {word} (w={wt})")
        else:
            print("  无")

        print("====================================\n")

    def top_words(self, n=10):
        print(f"\n========== 出现频率前 {n} 的单词 ==========")
        for i, (word, cnt) in enumerate(self.word_freq.most_common(n), start=1):
            print(f"{i:>2}. {word:<15} {cnt}")
        print("=========================================\n")

    def top_edges(self, n=10):
        edge_list = []
        for a, neighbors in self.graph.items():
            for b, wt in neighbors.items():
                edge_list.append((a, b, wt))

        edge_list.sort(key=lambda x: (-x[2], x[0], x[1]))

        print(f"\n========== 权重前 {n} 的边 ==========")
        for i, (a, b, wt) in enumerate(edge_list[:n], start=1):
            print(f"{i:>2}. {a} -> {b} (w={wt})")
        print("===================================\n")

    def find_path(self, start, end):
        """
        在有向图中找一条从 start 到 end 的路径（忽略权重，只看边是否存在）
        """
        start = start.lower()
        end = end.lower()

        if start not in self.word_freq:
            return None, f"起点单词 '{start}' 不在图中。"
        if end not in self.word_freq:
            return None, f"终点单词 '{end}' 不在图中。"

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path, None

            for nxt in self.graph.get(current, {}):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))

        return None, f"不存在从 '{start}' 到 '{end}' 的有向路径。"

    def print_path(self, start, end):
        path, err = self.find_path(start, end)
        print(f"\n========== 路径查询: {start} -> {end} ==========")
        if err:
            print(err)
        else:
            print("找到一条路径：")
            print(" -> ".join(path))
            print(f"路径长度（边数）: {len(path) - 1}")
        print("=============================================\n")
    
    def find_bridge_words(self, word1, word2):
        """
        查询 bridge words:
        若存在 word1 -> word3 且 word3 -> word2，则 word3 是桥接词
        """
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.word_freq or word2 not in self.word_freq:
            return None, f'No {word1} or {word2} in the graph!'

        bridge_words = []

        # 所有 word1 的后继节点
        next_words = self.graph.get(word1, {})

        for word3 in next_words:
            # 若存在 word3 -> word2
            if word2 in self.graph.get(word3, {}):
                bridge_words.append(word3)

        if not bridge_words:
            return [], f'No bridge words from {word1} to {word2}!'

        return sorted(bridge_words), None

    def print_bridge_words(self, word1, word2):
        bridge_words, err = self.find_bridge_words(word1, word2)

        if err:
            print("\n" + err + "\n")
            return

        if len(bridge_words) == 1:
            print(f"\nThe bridge words from {word1.lower()} to {word2.lower()} are: {bridge_words[0]}.\n")
        else:
            result = ", ".join(bridge_words[:-1]) + f", and {bridge_words[-1]}"
            print(f"\nThe bridge words from {word1.lower()} to {word2.lower()} are: {result}.\n")

    def print_bridge_words(self, word1, word2):
        bridge_words, err = self.find_bridge_words(word1, word2)

        if err:
            print("\n" + err + "\n")
            return

        if len(bridge_words) == 1:
            print(f"\nThe bridge words from {word1.lower()} to {word2.lower()} are: {bridge_words[0]}.\n")
        else:
            result = ", ".join(bridge_words[:-1]) + f", and {bridge_words[-1]}"
            print(f"\nThe bridge words from {word1.lower()} to {word2.lower()} are: {result}.\n")

    def get_bridge_words_list(self, word1, word2):
        """
        返回 word1 和 word2 之间所有 bridge words 的列表
        若 word1 或 word2 不在图中，返回 None
        若存在但没有 bridge word，返回空列表 []
        """
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.word_freq or word2 not in self.word_freq:
            return None

        bridge_words = []
        for word3 in self.graph.get(word1, {}):
            if word2 in self.graph.get(word3, {}):
                bridge_words.append(word3)

        return sorted(bridge_words)

    def generate_new_text(self, input_text):
        """
        根据输入的新文本生成插入 bridge word 后的新文本
        - 若两个相邻单词之间无 bridge word，则不插入
        - 若有多个，则随机选一个插入
        - 输出到屏幕上展示
        """
        # 提取原文本中的英文单词，尽量保留用户原始大小写用于输出
        original_words = re.findall(r"[A-Za-z]+", input_text)

        if not original_words:
            return ""

        result = [original_words[0]]

        for i in range(len(original_words) - 1):
            w1_original = original_words[i]
            w2_original = original_words[i + 1]

            w1 = w1_original.lower()
            w2 = w2_original.lower()

            bridge_words = self.get_bridge_words_list(w1, w2)

            # bridge_words is None 表示至少一个词不在图中
            # 按题意这里不报错，直接不插入，保持原样
            if bridge_words:
                chosen = random.choice(bridge_words)
                result.append(chosen)

            result.append(w2_original)

        return " ".join(result)
    
    def shortest_path(self, start, end):
        """
        使用 Dijkstra 计算 start 到 end 的最短路径
        路径长度定义为：路径上所有边权值之和最小
        返回:
            (path, dist, err)
            path: 最短路径节点列表
            dist: 最短路径总权值
            err : 出错信息或不可达信息
        """
        start = start.lower()
        end = end.lower()

        if start not in self.word_freq or end not in self.word_freq:
            return None, None, f"No {start} or {end} in the graph!"

        if start == end:
            return [start], 0, None

        # Dijkstra
        dist = {node: float("inf") for node in self.word_freq}
        prev = {}
        dist[start] = 0

        pq = [(0, start)]  # (当前距离, 当前节点)

        while pq:
            current_dist, u = heapq.heappop(pq)

            if current_dist > dist[u]:
                continue

            if u == end:
                break

            for v, w in self.graph.get(u, {}).items():
                new_dist = current_dist + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        if dist[end] == float("inf"):
            return None, None, f"No path from {start} to {end}!"

        # 回溯路径
        path = []
        cur = end
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.append(start)
        path.reverse()

        return path, dist[end], None

    def print_shortest_path(self, start, end):
        path, total_weight, err = self.shortest_path(start, end)

        print(f"\n========== 最短路径查询: {start} -> {end} ==========")
        if err:
            print(err)
        else:
            print("最短路径为：")
            print(" -> ".join(path))
            print(f"路径长度（所有边权值之和）: {total_weight}")
        print("===============================================\n")

    def show_shortest_path_on_graph(self, start, end):
        """
        将最短路径高亮显示在原图中，并弹窗展示
        依赖: networkx, matplotlib
        """
        path, total_weight, err = self.shortest_path(start, end)

        if err:
            print("\n" + err + "\n")
            return

        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n展示失败：缺少依赖库。")
            print("请先安装：")
            print("  pip install networkx matplotlib")
            print()
            # 至少仍然把路径打印出来
            self.print_shortest_path(start, end)
            return

        G = nx.DiGraph()

        for node, freq in self.word_freq.items():
            G.add_node(node, freq=freq)

        for a, neighbors in self.graph.items():
            for b, wt in neighbors.items():
                G.add_edge(a, b, weight=wt)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)

        # 所有节点
        default_node_sizes = [300 + 120 * G.nodes[n]["freq"] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=default_node_sizes)

        # 高亮路径节点
        path_nodes = set(path)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(path_nodes),
            node_size=[500 + 150 * G.nodes[n]["freq"] for n in path_nodes]
        )

        nx.draw_networkx_labels(G, pos, font_size=9)

        # 所有边先正常画
        all_edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, arrows=True, arrowstyle='->', arrowsize=15)

        # 路径边单独高亮
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            arrows=True,
            arrowstyle='->',
            arrowsize=20,
            width=3
        )

        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        title = f"Shortest Path: {' -> '.join(path)} | Length = {total_weight}"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        print("\n最短路径已在图中高亮展示：")
        print(" -> ".join(path))
        print(f"路径长度（所有边权值之和）: {total_weight}\n")

    def compute_pagerank(self, d=0.85, max_iter=100, tol=1e-6):
        """
        计算图中所有节点的 PageRank 值
        - d: 阻尼系数
        - max_iter: 最大迭代次数
        - tol: 收敛阈值
        返回:
            pr: dict, 每个节点对应的 PR 值
            iterations: 实际迭代次数
        说明:
        - 使用标准 PageRank
        - 仅按图结构计算，不使用边权
        - 对于出度为 0 的节点，将其 PR 值均分给所有节点
        """
        nodes = sorted(self.word_freq.keys())
        n = len(nodes)

        if n == 0:
            return {}, 0

        # 初始 PR
        pr = {node: 1.0 / n for node in nodes}

        # 预处理：每个节点的不同后继节点集合
        out_links = {}
        for node in nodes:
            out_links[node] = list(self.graph.get(node, {}).keys())

        for iteration in range(1, max_iter + 1):
            new_pr = {}

            for v in nodes:
                # 基础项
                rank_sum = 0.0

                for u in nodes:
                    u_out = out_links[u]

                    if len(u_out) == 0:
                        # 出度为 0，PR 均分给所有节点
                        rank_sum += pr[u] / n
                    else:
                        # 若存在 u -> v，则 u 对 v 有贡献
                        if v in u_out:
                            rank_sum += pr[u] / len(u_out)

                new_pr[v] = (1 - d) / n + d * rank_sum

            # 计算本轮变化量
            diff = sum(abs(new_pr[node] - pr[node]) for node in nodes)
            pr = new_pr

            if diff < tol:
                return pr, iteration

        return pr, max_iter

    def print_pagerank(self, d=0.85, max_iter=100, tol=1e-6):
        pr, iterations = self.compute_pagerank(d=d, max_iter=max_iter, tol=tol)

        print("\n========== PageRank 结果 ==========")
        print(f"阻尼系数 d = {d}")
        print(f"迭代次数 = {iterations}")
        print()

        sorted_pr = sorted(pr.items(), key=lambda x: (-x[1], x[0]))
        for i, (node, value) in enumerate(sorted_pr, start=1):
            print(f"{i:>2}. {node:<15} PR = {value:.6f}")

        print("==================================\n")

    def print_top_pagerank(self, top_n=10, d=0.85, max_iter=100, tol=1e-6):
        pr, iterations = self.compute_pagerank(d=d, max_iter=max_iter, tol=tol)
        sorted_pr = sorted(pr.items(), key=lambda x: (-x[1], x[0]))

        print(f"\n========== PageRank Top-{top_n} ==========")
        print(f"阻尼系数 d = {d}")
        print(f"迭代次数 = {iterations}")
        print()

        for i, (node, value) in enumerate(sorted_pr[:top_n], start=1):
            print(f"{i:>2}. {node:<15} PR = {value:.6f}")

        print("=========================================\n")

    def random_walk(self, output_file="random_walk.txt", interactive=True):
        """
        功能需求7：随机游走
        规则：
        1. 随机选择一个节点作为起点
        2. 每次从当前节点的所有出边中随机选择一条继续走
        3. 若将要走的边已经出现过，则仍将这一步加入结果，然后停止
        4. 若当前节点没有出边，则停止
        5. 用户可以在遍历过程中手动停止
        6. 将遍历的节点输出为文本，并写入磁盘文件
        返回：
            walk_text, stop_reason
        """
        nodes = list(self.word_freq.keys())
        if not nodes:
            return "", "图为空，无法进行随机游走。"

        current = random.choice(nodes)

        visited_edges = set()
        path_nodes = [current]
        path_edges = []

        while True:
            out_neighbors = list(self.graph.get(current, {}).keys())

            # 当前节点无出边，停止
            if not out_neighbors:
                stop_reason = f"节点 '{current}' 没有出边，随机游走结束。"
                break

            # 随机选一条出边
            next_node = random.choice(out_neighbors)
            edge = (current, next_node)

            # 记录这条边和到达的节点
            path_edges.append(edge)
            path_nodes.append(next_node)

            # 如果这条边重复，停止
            if edge in visited_edges:
                stop_reason = f"出现第一条重复边 {current} -> {next_node}，随机游走结束。"
                break

            visited_edges.add(edge)
            current = next_node

            # 用户手动停止
            if interactive:
                user_input = input("按回车继续随机游走，输入 q 停止：").strip().lower()
                if user_input == "q":
                    stop_reason = "用户手动停止随机游走。"
                    break

        walk_text = " ".join(path_nodes)

        # 写入文件
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(walk_text + "\n")
        except Exception as e:
            stop_reason += f"\n写入文件失败：{e}"

        return walk_text, stop_reason

    def print_random_walk(self, output_file="random_walk.txt", interactive=True):
        walk_text, stop_reason = self.random_walk(output_file=output_file, interactive=interactive)

        print("\n========== 随机游走结果 ==========")
        print(walk_text)
        print()
        print(stop_reason)
        print(f"游走结果已写入文件：{output_file}")
        print("==================================\n")

    def save_graph_image(self, output_file="graph.png"):
        """
        功能需求2（可选）：
        将生成的有向图保存为图片文件
        依赖: networkx, matplotlib
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n保存图片失败：缺少依赖库。")
            print("请先安装：")
            print("  pip install networkx matplotlib")
            print()
            return

        G = nx.DiGraph()

        for node, freq in self.word_freq.items():
            G.add_node(node, freq=freq)

        for a, neighbors in self.graph.items():
            for b, wt in neighbors.items():
                G.add_edge(a, b, weight=wt)

        if len(G.nodes) == 0:
            print("图为空，无法保存。")
            return

        plt.figure(figsize=(12, 8))

        # 布局
        pos = nx.spring_layout(G, seed=42)

        # 节点大小根据词频变化
        node_sizes = [300 + 120 * G.nodes[n]["freq"] for n in G.nodes()]

        # 绘制节点、边、标签
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
        nx.draw_networkx_labels(G, pos, font_size=9)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=15)

        # 绘制边权重
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Directed Word Graph")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"\n有向图图片已保存到: {output_file}\n")


def read_text_file(file_path):
    """
    尝试多种常见编码读取文本文件
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法读取文件，最后一次错误信息：{last_error}")


def normalize_text_to_words(text):
    """
    规则：
    - 仅保留 A-Z / a-z
    - 不区分大小写，统一转小写
    - 换行、标点、数字、其他字符都作为分隔处理
    """
    words = re.findall(r"[A-Za-z]+", text.lower())
    return words


def build_graph_from_file(file_path):
    raw_text = read_text_file(file_path)
    words = normalize_text_to_words(raw_text)

    if not words:
        raise ValueError("文件中没有可用的英文单词。")

    return WordGraph(words), raw_text, words


def interactive_menu(wg, raw_text, words):
    while True:
        print("请选择操作：")
        print("1. 查看图摘要")
        print("2. 以清晰格式展示有向图（CLI）")
        print("3. 以紧凑邻接表展示图")
        print("4. 查询某个单词的信息")
        print("5. 查询两个单词的桥接词")
        print("6. 根据 bridge word 生成新文本")
        print("7. 计算两个单词之间的最短路径")
        print("8. 在图中高亮展示最短路径")
        print("9. 计算全部节点的 PageRank")
        print("10. 查看 PageRank Top-N")
        print("11. 随机游走")
        print("12. 查看高频单词 Top-N")
        print("13. 查看高权重边 Top-N")
        print("14. 查询两个单词之间的一条普通路径")
        print("15. 查看清洗后的文本前100个单词")
        print("16. 保存有向图为图片文件")
        print("0. 退出程序")

        choice = input("请输入选项编号：").strip()

        if choice == "1":
            wg.print_summary()

        elif choice == "2":
            wg.print_graph()

        elif choice == "3":
            wg.print_graph_compact()

        elif choice == "4":
            word = input("请输入要查询的单词：").strip()
            wg.query_word(word)

        elif choice == "5":
            word1 = input("请输入 word1：").strip()
            word2 = input("请输入 word2：").strip()
            wg.print_bridge_words(word1, word2)

        elif choice == "6":
            input_text = input("请输入一行新文本：\n").strip()
            new_text = wg.generate_new_text(input_text)
            print("\n生成的新文本为：")
            print(new_text)
            print()

        elif choice == "7":
            start = input("请输入起点单词：").strip()
            end = input("请输入终点单词：").strip()
            wg.print_shortest_path(start, end)

        elif choice == "8":
            start = input("请输入起点单词：").strip()
            end = input("请输入终点单词：").strip()
            wg.show_shortest_path_on_graph(start, end)

        elif choice == "9":
            try:
                d = float(input("请输入阻尼系数 d（默认 0.85）：").strip() or "0.85")
            except ValueError:
                d = 0.85
            wg.print_pagerank(d=d)

        elif choice == "10":
            try:
                top_n = int(input("请输入 Top-N（默认 10）：").strip() or "10")
            except ValueError:
                top_n = 10

            try:
                d = float(input("请输入阻尼系数 d（默认 0.85）：").strip() or "0.85")
            except ValueError:
                d = 0.85

            wg.print_top_pagerank(top_n=top_n, d=d)

        elif choice == "11":
            output_file = input("请输入随机游走结果输出文件名（默认 random_walk.txt）：").strip()
            if not output_file:
                output_file = "random_walk.txt"

            mode = input("是否启用手动逐步停止模式？(y/n，默认 y)：").strip().lower()
            interactive = (mode != "n")

            wg.print_random_walk(output_file=output_file, interactive=interactive)

        elif choice == "12":
            try:
                n = int(input("请输入 N：").strip())
            except ValueError:
                n = 10
            wg.top_words(n)

        elif choice == "13":
            try:
                n = int(input("请输入 N：").strip())
            except ValueError:
                n = 10
            wg.top_edges(n)

        elif choice == "14":
            start = input("请输入起点单词：").strip()
            end = input("请输入终点单词：").strip()
            wg.print_path(start, end)

        elif choice == "15":
            print("\n========== 清洗后的文本预览 ==========")
            preview = " ".join(words[:100])
            print(preview if preview else "[空]")
            print("====================================\n")

        elif choice == "16":
            output_file = input("请输入输出图片文件名（默认 graph.png）：").strip()
            if not output_file:
                output_file = "graph.png"
            wg.save_graph_image(output_file)

        elif choice == "0":
            print("程序结束。")
            break

        else:
            print("无效选项，请重新输入。\n")



def main():
    parser = argparse.ArgumentParser(description="从文本文件构建单词有向图")
    parser.add_argument("file", nargs="?", help="文本文件路径")
    args = parser.parse_args()

    file_path = args.file
    if not file_path:
        file_path = input("请输入文本文件路径：").strip()

    if not os.path.isfile(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return

    try:
        wg, raw_text, words = build_graph_from_file(file_path)
    except Exception as e:
        print(f"读取或处理文件失败：{e}")
        return

    print("\n文件读取成功，已生成有向图。")
    wg.print_summary()

    print("下面先用 CLI 清晰展示生成的有向图：")
    wg.print_graph()

    interactive_menu(wg, raw_text, words)


if __name__ == "__main__":
    main()