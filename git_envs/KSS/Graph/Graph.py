# coding: utf-8
# create by tongshiwei on 2018/10/22

import os
import networkx as nx


def load_graph(filename, graph_nodes_num=None):
    graph = nx.DiGraph()

    if graph_nodes_num is not None:
        graph.add_nodes_from(range(graph_nodes_num))

    with open(filename) as f:
        edges = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

    graph.add_edges_from(edges)
    return graph


def load_id2idx(filename):
    id2idx = {}
    with open(filename) as f:
        for line in f:
            if line.strip():
                vid, idx = line.strip().split(',')
                id2idx[vid] = int(idx)

    return id2idx


def load_idx2id(filename):
    idx2id = {}
    with open(filename) as f:
        for line in f:
            if line.strip():
                vid, idx = line.strip().split(',')
                idx2id[int(idx)] = vid

    return idx2id


def bfs(graph, mastery, pnode, hop, candidates, soft_candidates, visit_nodes=None, visit_threshold=1,
        allow_shortcut=True):
    """

    Parameters
    ----------
    graph: nx.Digraph
    mastery
    pnode
    hop
    candidates: set()
    soft_candidates: set()
    visit_nodes
    visit_threshold

    Returns
    -------

    """
    assert hop >= 0
    if visit_nodes and visit_nodes.get(pnode, 0) >= visit_threshold:
        return

    # 当前节点没掌握则加入候选
    if allow_shortcut is False or mastery[pnode] < 0.5:
        candidates.add(pnode)
    else:
        soft_candidates.add(pnode)

    if hop == 0:
        return

    # 向前搜索
    # 该节点之前hop跳内均执行本操作
    for node in list(graph.predecessors(pnode)):
        if allow_shortcut is False or mastery[node] < 0.5:
            bfs(
                graph=graph,
                mastery=mastery,
                pnode=node,
                hop=hop - 1,
                candidates=candidates,
                soft_candidates=soft_candidates,
                visit_nodes=visit_nodes,
                visit_threshold=visit_threshold,
                allow_shortcut=allow_shortcut,
            )

    # 向后搜索
    # 该节点的后继若没掌握则加入候选
    for node in list(graph.successors(pnode)):
        if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
            continue
        if allow_shortcut is False or mastery[node] < 0.5:
            candidates.add(node)
        else:
            soft_candidates.add(node)


def graph_candidate(graph, mastery, pnode, visit_nodes=None, visit_threshold=1, allow_shortcut=True, no_pre=None,
                    connected_graph=None, target=None, legal_candidates=None, path_table=None):
    """

    Parameters
    ----------
    graph: nx.Digraph
    mastery: list(float)
    pnode: None or int
    visit_nodes: None or dict
    visit_threshold: int
    allow_shortcut: bool
    no_pre: set
    connected_graph: dict
    target: set or list
    legal_candidates: set or None
    path_table: dict or None

    Returns
    -------

    """
    if mastery is None:
        allow_shortcut = False

    # select candidates
    candidates = []
    soft_candidates = []

    if allow_shortcut is True:
        # 允许通过捷径绕过已掌握的点

        # 在已有前驱节点前提下，如果当前节点已经掌握，那么开始学习它的后继未掌握节点
        if pnode is not None and mastery[pnode] >= 0.5:
            for candidate in list(graph.successors(pnode)):
                if visit_nodes and visit_nodes.get(candidate, 0) >= visit_threshold:
                    continue
                if mastery[candidate] < 0.5:
                    candidates.append(candidate)
                else:
                    soft_candidates.append(candidate)
            if candidates:
                return candidates, soft_candidates

        # 否则(即当前节点未掌握), 选取其2跳前驱点及所有前驱点的后继点（未掌握的）作为候选集
        elif pnode is not None:
            _candidates = set()
            _soft_candidates = set()
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 2, _candidates, _soft_candidates, visit_nodes, visit_threshold,
                    allow_shortcut)
            return list(_candidates) + [pnode], list(_soft_candidates)

        # 如果前两种方法都没有选取到候选集，那么进行重新选取
        for node in graph.nodes:
            if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
                # 当前结点频繁访问
                continue

            if mastery[node] >= 0.5:
                # 当前结点已掌握，跳过
                soft_candidates.append(node)
                continue

            # 当前结点未掌握，且其前置点都掌握了的情况下，加入候选集
            pre_nodes = list(graph.predecessors(node))
            for n in pre_nodes:
                pre_mastery = mastery[n]
                if pre_mastery < 0.5:
                    soft_candidates.append(node)
                    break
            else:
                candidates.append(node)
    else:
        # allow_shortcut is False
        # 不允许通过捷径绕过已掌握的点
        candidates = set()
        soft_candidates = set()
        if pnode is not None:
            # 加入所有后继点
            candidates = set(list(graph.successors(pnode)))

            if not graph.predecessors(pnode) or not graph.successors(pnode):
                # 没有前驱点 或 没有后继点: 没有前驱点的节点作为候选
                candidates = set(no_pre)

            # 选取其2跳前驱点及所有前驱点的后继点
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 1, candidates, soft_candidates, visit_nodes, visit_threshold, allow_shortcut)

            # 避免死循环
            if candidates:
                candidates.add(pnode)

            # 频繁访问节点过滤
            if visit_nodes:
                candidates -= set([node for node, count in visit_nodes.items() if count >= visit_threshold])

            candidates = list(candidates)

    if not candidates:
        # 规则没有选取到合适候选集
        candidates = list(graph.nodes)
        soft_candidates = list()

    if connected_graph is not None and pnode is not None:
        # 保证候选集和pnode在同一个连通子图内
        candidates = list(set(candidates) & connected_graph[pnode])

    if target is not None and legal_candidates is not None:
        assert target
        # 保证节点可达目标点
        _candidates = set(candidates) - legal_candidates
        for candidate in _candidates:
            if candidate in legal_candidates:
                continue
            for t in target:
                if path_table is not None:
                    if t in path_table[candidate]:
                        legal_tag = True
                    else:
                        legal_tag = False
                else:
                    legal_tag = nx.has_path(graph, candidate, t)
                if legal_tag is True:
                    legal_candidates.add(candidate)
                    break
        candidates = set(candidates) & legal_candidates
        if not candidates:
            candidates = target

    return list(candidates), list(soft_candidates)

# def graph_candidate(graph, mastery, pnode, visit_nodes=None, visit_threshold=1, allow_shortcut=True, no_pre=None,
#                     connected_graph=None, target=None, legal_candidates=None, path_table=None):
#     """
#
#     Parameters
#     ----------
#     graph: nx.Digraph
#     mastery: list(float)
#     pnode: None or int
#     visit_nodes: None or dict
#     visit_threshold: int
#     allow_shortcut: bool
#     no_pre: set
#     connected_graph: dict
#     target: set or list
#     legal_candidates: set or None
#     path_table: dict or None
#
#     Returns
#     -------
#
#     """
#     if mastery is None:
#         allow_shortcut = False
#
#     # select candidates
#     candidates = []
#     soft_candidates = []
#
#     if allow_shortcut is True:
#         # 允许通过捷径绕过已掌握的点
#
#         # 在已有前驱节点前提下，如果当前节点已经掌握，那么开始学习它的后继未掌握节点
#         if pnode is not None and mastery[pnode] >= 0.5:
#             for candidate in list(graph.successors(pnode)):
#                 if visit_nodes and visit_nodes.get(candidate, 0) >= visit_threshold:
#                     continue
#                 if mastery[candidate] < 0.5:
#                     candidates.append(candidate)
#                 else:
#                     soft_candidates.append(candidate)
#             if candidates:
#                 return candidates, soft_candidates
#
#         # 否则(即当前节点未掌握), 选取其2跳前驱点及所有前驱点的后继点（未掌握的）作为候选集
#         elif pnode is not None:
#             _candidates = set()
#             _soft_candidates = set()
#             for node in list(graph.predecessors(pnode)):
#                 bfs(graph, mastery, node, 2, _candidates, _soft_candidates, visit_nodes, visit_threshold,
#                     allow_shortcut)
#             return list(_candidates) + [pnode], list(_soft_candidates)
#
#         # 如果前两种方法都没有选取到候选集，那么进行重新选取
#         for node in graph.nodes:
#             if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
#                 # 当前结点频繁访问
#                 continue
#
#             if mastery[node] >= 0.5:
#                 # 当前结点已掌握，跳过
#                 soft_candidates.append(node)
#                 continue
#
#             # 当前结点未掌握，且其前置点都掌握了的情况下，加入候选集
#             pre_nodes = list(graph.predecessors(node))
#             for n in pre_nodes:
#                 pre_mastery = mastery[n]
#                 if pre_mastery < 0.5:
#                     soft_candidates.append(node)
#                     break
#             else:
#                 candidates.append(node)
#     else:
#         # allow_shortcut is False
#         # 不允许通过捷径绕过已掌握的点
#         candidates = set()
#         soft_candidates = set()
#         if pnode is not None:
#             for suc_node1 in graph.successors(pnode):
#                 for suc_node2 in graph.successors(suc_node1):
#                     candidates.add(suc_node2)
#                 candidates.add(suc_node1)
#
#             for pre_node1 in graph.predecessors(pnode):
#                 for pre_node2 in graph.predecessors(pre_node1):
#                     candidates.add(pre_node2)
#                     for suc_pre_node2 in graph.successors(pre_node2):
#                         candidates.add(suc_pre_node2)
#
#                 for suc_pre_node1 in graph.successors(pre_node1):
#                     candidates.add(suc_pre_node1)
#                 candidates.add(pre_node1)
#             # 加入所有后继点
#             # candidates = set(list(graph.successors(pnode)))
#             #
#             # if not graph.predecessors(pnode) or not graph.successors(pnode):
#             #     # 没有前驱点 或 没有后继点: 没有前驱点的节点作为候选
#             #     candidates = set(no_pre)
#             #
#             # # 选取其2跳前驱点及所有前驱点的后继点
#             # for node in list(graph.predecessors(pnode)):
#             #     bfs(graph, mastery, node, 1, candidates, soft_candidates, visit_nodes, visit_threshold, allow_shortcut)
#             #
#             # # 避免死循环
#             # if candidates:
#             #     candidates.add(pnode)
#             #
#             # # 频繁访问节点过滤
#             # if visit_nodes:
#             #     candidates -= set([node for node, count in visit_nodes.items() if count >= visit_threshold])
#             #
#             # candidates = list(candidates)
#
#     if not candidates:
#         # 规则没有选取到合适候选集
#         candidates = list(graph.nodes)
#         soft_candidates = list()
#
#     if connected_graph is not None and pnode is not None:
#         # 保证候选集和pnode在同一个连通子图内
#         candidates = list(set(candidates) & connected_graph[pnode])
#
#     if target is not None and legal_candidates is not None:
#         assert target
#         # 保证节点可达目标点
#         _candidates = set(candidates) - legal_candidates
#         for candidate in _candidates:
#             if candidate in legal_candidates:
#                 continue
#             for t in target:
#                 if path_table is not None:
#                     if t in path_table[candidate]:
#                         legal_tag = True
#                     else:
#                         legal_tag = False
#                 else:
#                     legal_tag = nx.has_path(graph, candidate, t)
#                 if legal_tag is True:
#                     legal_candidates.add(candidate)
#                     break
#         candidates = set(candidates) & legal_candidates
#         if not candidates:
#             candidates = target
#
#     return list(candidates), list(soft_candidates)

class Graph(object):
    def __init__(self, dataset=None, graph_nodes_num=None, disable=False):
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../data/%s/data/graph_edges.idx" % dataset))
        id2idx_filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../data/%s/data/vertex_id2idx" % dataset))

        self._id2idx = load_id2idx(id2idx_filename) if id2idx_filename is not None else None
        self._idx2id = load_idx2id(id2idx_filename) if id2idx_filename is not None else None

        _graph_nodes_num = max(self._id2idx.values())
        assert _graph_nodes_num == max(self._idx2id.keys())

        self.graph = load_graph(filename, graph_nodes_num if graph_nodes_num is not None else _graph_nodes_num)

        self.no_pre = [node for node in self.nodes if not list(self.graph.predecessors(node))]

        self.disable = disable

        self.connected_graph = {}

        self.path_table = nx.shortest_path(self.graph)

        self.initial_connected_graph()

    def __call__(self, mastery, pnode, visit_nodes=None, visit_threshold=1, allow_shortcut=True, target=None,
                 legal_candidates=None):
        if self.disable is False:
            return graph_candidate(self.graph, mastery, pnode, visit_nodes, visit_threshold, allow_shortcut,
                                   self.no_pre, self.connected_graph, target, legal_candidates, self.path_table)
        else:
            return list(self.nodes), []

    @property
    def nodes(self):
        return self.graph.nodes

    def id2idx(self, vid):
        return self._id2idx[vid]

    def idx2id(self, idx):
        return self._idx2id[idx]

    def predecessors(self, idx):
        return list(self.graph.predecessors(idx))

    def successors(self, idx):
        return list(self.graph.successors(idx))

    def parents(self, idx):
        return list(self.predecessors(idx))

    def grandparents(self, idx):
        gp = []

        for node in self.parents(idx):
            gp += self.predecessors(node)

        return gp

    def parents_siblings(self, idx):
        ps = []

        for node in self.grandparents(idx):
            ps += self.graph.successors(node)

        return ps

    def initial_connected_graph(self):
        for node in self.graph.nodes:
            if node in self.connected_graph:
                continue
            else:
                queue = [node]
                _connected_graph = set()
                while queue:
                    visit = queue.pop()
                    if visit not in _connected_graph:
                        _connected_graph.add(visit)
                        queue.extend(self.predecessors(visit))
                        queue.extend(self.successors(visit))
                for node in _connected_graph:
                    self.connected_graph[node] = _connected_graph

    def spotlight(self, node, level):
        if type(level) is int:
            level = (level, level)
        spot_nodes = set()

        pre_level, suc_level = level

        def pre_dfs(node, _level):
            spot_nodes.add(node)
            if _level <= 0:
                return
            else:
                for _node in self.predecessors(node):
                    pre_dfs(_node, _level - 1)

        def suc_dfs(node, _level):
            spot_nodes.add(node)
            if _level <= 0:
                return
            else:
                for _node in self.successors(node):
                    suc_dfs(_node, _level - 1)

        pre_dfs(node, pre_level)
        suc_dfs(node, suc_level)

        spot_edges = [edge for edge in self.graph.edges if edge[0] in spot_nodes and edge[1] in spot_nodes]

        return list(spot_nodes), spot_edges

    def younger(self, a, b):
        queue = list(self.successors(a))
        _younger = set()
        while queue:
            visit = queue.pop()
            if visit not in _younger:
                _younger.add(visit)
                queue.extend(self.successors(visit))
        if b in _younger:
            return True
        else:
            return False

    def elder(self, a, b):
        return self.younger(b, a)

    def spot_path(self, path):
        pnode = path[0]
        spot_nodes = set()
        for p in path[1:]:
            if pnode in spot_nodes and p in spot_nodes:
                pnode = p
                continue
            else:
                for simple_path in nx.all_simple_paths(self.graph, pnode, p):
                    spot_nodes.update(set(simple_path))
                pnode = p
        spot_edges = [edge for edge in self.graph.edges if edge[0] in spot_nodes and edge[1] in spot_nodes]

        return list(spot_nodes), spot_edges


if __name__ == '__main__':
    graph = Graph(dataset="junyi")
    # idx = graph.id2idx("parabola_intuition_1")
    idx = 630


    # print(graph.idx2id(743))
    # print(len(graph.nodes), len(graph.graph.edges))
    # print(graph.predecessors(630))

    # for node in graph.nodes:
    #     if "quadra" in graph.idx2id(node):
    #         print(node, graph.idx2id(node))

    def no_link(nodes):
        for idx, node in enumerate(nodes):
            for i in range(idx + 1, len(nodes)):
                if nodes[i] in graph.path_table[node] or node in graph.path_table[nodes[i]]:
                    return False
        return True


    def in_any(string, elements):
        for elem in elements:
            if string in graph.idx2id(elem):
                return True
        return False


    def all_shorter(length, elements):
        for elem in elements:
            if len(graph.idx2id(elem)) > length:
                return False
        return True


    # for n1 in graph.nodes:
    #     for n2 in graph.nodes:
    #         if no_link([n1, n2]) and set(
    #                 graph.path_table[n1].keys()) & set(graph.path_table[n2].keys()) and any(
    #                 [list(graph.predecessors(n1)), list(graph.predecessors(n2))]):
    #             for n3 in set(graph.path_table[n1].keys()) & set(graph.path_table[n2].keys()):
    #                 if graph.successors(n3):
    #                     for n4 in graph.successors(n3):
    #                         for n5 in graph.predecessors(n4):
    #                             if no_link([n1, n2, n5]) and in_any("multi", [n3]) and in_any("add",
    #                                                                                           [n1, n2]) and all_shorter(
    #                                 20, [n1, n2, n3, n4, n5]):
    #                                 print(n1, n2, n3, n4, n5)
    #                                 exit(0)


    path = [664, 241, 174, 174]
    # assert len(set(path)) == 14, len(path)
    for node in path:
        print(node, graph.idx2id(node))

    # print(graph.spotlight(630, (2, 0)))
    # print([graph.idx2id(node) for node in graph.graph.predecessors(idx)])

    # print(len(graph.graph.edges))
    # print(len(graph.no_pre))
