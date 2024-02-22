# coding:utf-8
# created by tongshiwei on 2018/11/7
import os
import networkx as nx

# todo 检查网络节点是否反了
def load_graph(filename, graph_nodes_num=None):
    graph = nx.DiGraph()

    if graph_nodes_num is not None:
        graph.add_nodes_from(range(graph_nodes_num))

    with open(filename) as f:
        edges = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

    graph.add_edges_from(edges)
    # print(list(graph.predecessors(1)))

    # import matplotlib.pyplot as plt
    # nx.draw_networkx(graph)
    # plt.show()
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


class GraphCheck(object):
    def __init__(self, filename=None, dataset=None, graph_nodes_num=None, id2idx_filename=None):
        if filename:
            self.graph = load_graph(filename)
        elif dataset:
            filename = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../data/%s/data/graph_edges.idx" % dataset))
            id2idx_filename = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../data/%s/data/vertex_id2idx" % dataset))
            self.graph = load_graph(filename, graph_nodes_num)
        else:
            raise AssertionError()

        self._id2idx = load_id2idx(id2idx_filename) if id2idx_filename is not None else None
        self._idx2id = load_idx2id(id2idx_filename) if id2idx_filename is not None else None

    def id2idx(self, vid):
        return self._id2idx[vid]

    def idx2id(self, idx):
        return self._idx2id[idx]


if __name__ == '__main__':
    # graph = GraphCheck(dataset="junyi_80")
    # idx = graph.id2idx("greatest_common_divisor")
    # # idx = 644
    # print(graph.idx2id(idx))
    # print([graph.idx2id(node) for node in graph.graph.successors(idx)])
    # print([graph.idx2id(node) for node in graph.graph.predecessors(idx)])

    graph = nx.DiGraph()

    graph.add_edges_from([(0, 1), (1, 2)])

    print(list(graph.predecessors(1)))
    print(list(graph.predecessors(2)))
