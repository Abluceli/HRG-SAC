import numpy as np


class Sum_Tree(object):
    def __init__(self, capacity):
        """
        capacity = 5，设置经验池大小
        tree = [0,1,2,3,4,5,6,7,8,9,10,11,12] 8-12存放叶子结点p值，1-7存放父节点、根节点p值的和，0存放树节点的数量
        data = [0,1,2,3,4,5] 1-5存放数据， 0存放capacity
        Tree structure and array storage:
        Tree index:
                    1         -> storing priority sum
              /          \ 
             2            3
            / \          / \
          4     5       6   7
         / \   / \     / \  / \
        8   9 10   11 12                   -> storing priority for transitions
        """
        assert capacity > 0, 'capacity must larger than zero'
        self.now = 0
        self.capacity = capacity
        self.parent_node_count = self.get_parent_node_count(capacity)
        # print(self.parent_node_count)
        self.tree = np.zeros(self.parent_node_count + capacity + 1)
        self.tree[0] = len(self.tree) - 1
        self.data = np.zeros(capacity + 1, dtype=object)
        self.data[0] = capacity

    def add(self, p, data):
        """
        p : property
        data : [s, a, r, s_, done]
        """
        idx = self.now + 1
        self.data[idx] = data
        tree_index = idx + self.parent_node_count
        self._updatetree(tree_index, p)
        if idx >= self.capacity:
            self.now = 0
        else:
            self.now = idx

    def add_batch(self, p, data):
        """
        p : property
        data : [s, a, r, s_, done]
        """
        num = len(data)
        idx = (np.arange(num) + self.now) % self.capacity + 1
        self.data[idx] = data
        tree_index = idx + self.parent_node_count
        self._updatetree_batch(tree_index, p)
        if idx[-1] >= self.capacity:
            self.now = 0
        else:
            self.now = idx[-1]

    def _updatetree(self, tree_index, p):
        diff = p - self.tree[tree_index]
        self._propagate(tree_index, diff)
        self.tree[tree_index] = p

    def _updatetree_batch(self, tree_index, p):
        diff = p - self.tree[tree_index]
        sort_index = np.argsort(tree_index)
        tree_index = np.sort(tree_index)
        diff = diff[sort_index]
        self._propagate_batch(tree_index, diff)
        self.tree[tree_index] = p

    def _propagate(self, tree_index, diff):
        parent = tree_index // 2
        self.tree[parent] += diff
        if parent != 1:
            self._propagate(parent, diff)

    def _propagate_batch(self, tree_index, diff):
        parent = tree_index // 2
        _parent, idx1, count = np.unique(parent, return_index=True, return_counts=True)
        _, idx2 = np.unique(parent[::-1], return_index=True)
        diff = (diff[len(diff) - 1 - idx2] + diff[idx1]) * count / 2
        self.tree[_parent] += diff
        if (_parent != 1).all():
            self._propagate_batch(_parent, diff)

    def get(self, seg_p_total):
        """
        seg_p_total : The value of priority to sample
        """
        tree_index = self._retrieve(1, seg_p_total)
        data_index = tree_index - self.parent_node_count
        return (tree_index, data_index, self.tree[tree_index], self.data[data_index])

    def get_batch(self, ps):
        assert isinstance(ps, (list, np.ndarray))
        tidx, didx, p, d = zip(*[self.get(i) for i in ps])
        tidx, didx, p, d = map(np.asarray, [tidx, didx, p, d])
        d = [np.asarray(e) for e in zip(*d)]    # [[s, a], [s, a]] => [[s, s], [a, a]]
        return (tidx, didx, p, d)

    def get_batch_parallel(self, ps):
        assert isinstance(ps, (list, np.ndarray))
        init_idx = np.full(len(ps), 1)
        tidx = self._retrieve_batch(init_idx, ps)
        didx = tidx - self.parent_node_count
        p = self.tree[tidx]
        d = self.data[didx]
        tidx, didx, p, d = map(np.asarray, [tidx, didx, p, d])
        d = [np.asarray(e) for e in zip(*d)]    # [[s, a], [s, a]] => [[s, s], [a, a]]
        return (tidx, didx, p, d)

    def _retrieve(self, tree_index, seg_p_total):
        left = 2 * tree_index
        right = left + 1
        # if index 0 is the root node
        # left = 2 * tree_index + 1
        # right = 2 * (tree_index + 1)
        if left >= self.tree[0]:
            return tree_index
        return self._retrieve(left, seg_p_total) if seg_p_total <= self.tree[left] else self._retrieve(right, seg_p_total - self.tree[left])

    def _retrieve_batch(self, tree_index, seg_p_total):
        left = 2 * tree_index
        right = left + 1
        if (left >= self.tree[0]).all():
            return tree_index
        index = np.where(self.tree[left] >= seg_p_total, left, 0) + np.where(self.tree[left] < seg_p_total, right, 0)
        seg_p_total = np.where(self.tree[left] >= seg_p_total, seg_p_total, 0) + np.where(self.tree[left] < seg_p_total, seg_p_total - self.tree[left], 0)
        return self._retrieve_batch(index, seg_p_total)

    def pp(self):
        print(self.tree, self.data)

    @property
    def total(self):
        return self.tree[1]

    def get_parent_node_count(self, capacity):
        i = 0
        while True:
            if pow(2, i) < capacity <= pow(2, i + 1):
                return pow(2, i + 1) - 1
            i += 1


if __name__ == "__main__":
    from time import time
    x = 0
    t = 1000
    for i in range(t):
        tree = Sum_Tree(524288)
        a = np.arange(50000)
        b = np.zeros_like(a)
        start = time()
        tree.add_batch(b, a)
        x += time() - start
    print(x / t)
