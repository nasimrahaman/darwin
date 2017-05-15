cimport numpy as np
import numpy as np

ctypedef np.float32_t dtype_t
ctypedef np.uint8_t i_type
ctypedef unsigned long ULong

from libc.stdlib cimport malloc, free

# union find code originally from a
# adjusted to model seeds
import numpy as np

class Watershredder(object):
    ''' This gnarly class executes the learned Watershed algorithm based on union find datastructure'''

    def __init__(self, image_shape):
        self.image_shape = image_shape
        # self.vis = visualization.Vizualizer(options)
        self.uf = UnionFind(self.image_shape)
        self.label_image = None
        self.label_image = None

    def execute_kruskal_updates(self, weights, iterations=0):
        '''applies <iteration> number of kruskal update steps on both union find data structures'''
        sorted_actions = np.argsort(weights.flatten())
        self.uf.kruskal_iteration(sorted_actions, iterations)

    def get_label_image(self):
        return self.uf.get_flat_label_image().reshape(self.image_shape)

    def get_display_label_image(self):
        return self.uf.get_flat_label_image_only_merged_pixels().reshape(self.image_shape)




cdef class UnionFind:
    cdef int n_points
    cdef int n_dims
    cdef int * parent
    cdef int * rank
    cdef int _n_sets
    # cdef np.ndarray[i_type, ndim=1] seed
    cdef int * seed
    cdef int * shape
    cdef int * region_size
    cdef int * strides
    cdef int * actions
    cdef list seeded_labels

    cdef int action_counter 

    def __cinit__(self, image_shape):
        cdef int stride = 1
        cdef int n_points = 1
        cdef int ndims = len(image_shape)
        self.n_dims = ndims
        self.action_counter = 0

        self.strides = <int *>malloc(ndims * sizeof(int))

        for direction_index in range(ndims):
            self.strides[direction_index] = stride
            n_points *= image_shape[direction_index]
            stride *= image_shape[ndims-1-direction_index]
        
        self.n_points = n_points
        self.parent = <int *> malloc(n_points * sizeof(int))
        self.rank = <int *> malloc(n_points * sizeof(int))
        self.region_size = <int *> malloc(n_points * sizeof(int))
        self.shape = <int *>malloc(ndims * sizeof(int))
        self.seed = <int *> malloc(n_points * sizeof(int))
        self.seeded_labels = []

        self.actions = <int *>malloc(self.n_points * (self.n_dims + 1) * sizeof(int))
        
        for i in range(ndims):
            self.shape[i] = image_shape[i]
        self.clear()

    def __dealloc__(self):
        free(self.parent)
        free(self.rank)
        free(self.region_size)
        free(self.seed)
        free(self.shape)
        free(self.strides)

    cdef int _find(self, int i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

    def _check_bounds(self, int i, int k):
        if self.n_dims == 2:
            if k == 0:
                return i % self.shape[1] != self.shape[1] -1
            if k == 1:
                return i / self.shape[1] % self.shape[0] != self.shape[0] -1
        elif self.n_dims == 3:
            if k == 2:
                return (i % self.shape[2] != self.shape[2] -1)
            if k == 1:
                return (i / (self.shape[2])) % (self.shape[1]) != self.shape[1] -1
            if k == 0:
                return (i % ((self.shape[2]) * (self.shape[1]))) != self.shape[0] -1
        else:
            return False

    def clear(self):
        cdef int i
        for i in range(self.n_points):
            self.seed[i] = 0
            self.rank[i] = 0
            self.region_size[i] = 1
            self.parent[i] = i

        for i in range(self.n_points * (self.n_dims + 1)):
            self.actions[i] = 0


    def find(self, int i):
        if (i < 0) or (i > self.n_points):
            raise ValueError("Out of bounds index.")
        return self._find(i)

    def add_seed(self, int i):
        root_i = self._find(i)
        r = not self.seed[root_i]
        self.seed[root_i] = True
        return r

    def get_flat_label_image_only_merged_pixels(self):
        cdef np.ndarray[long, ndim=1] label
        label =  np.zeros(self.n_points, dtype=long)
        cdef int i, size, root_i
        for i in range(self.n_points):
            root_i = self._find(i)
            size = self.region_size[root_i]
            if size > 1 or self.seed[i] == 1:
                label[i] = root_i+1
            else:
                label[i] = 0
        return label

    def get_flat_label_image(self):
        cdef np.ndarray[long, ndim=1] label
        label =  np.zeros(self.n_points, dtype=long)
        cdef int i, size, root_i
        for i in range(self.n_points):
            root_i = self._find(i)
            size = self.region_size[root_i]
            label[i] = root_i+1
        return label

    def get_flat_label_projection(self):
        ''' this function generates an id-invariant projection of the current segmentation'''
        cdef np.ndarray[long, ndim=1] region_projection
        region_projection =  np.zeros(self.n_points * 3, dtype=long)
        cdef int i, d, root_i
        # # index: 0: seeded regions
        # # index: 1: boundary pixel
        # # index: 2: region size
        for i in range(self.n_points):
            root_i = self._find(i)
            if self.seed[self._find(i)] == 1:
                region_projection[3*i+0] = 1
            
            for d in range(self.n_dims):
                if self._check_bounds(i, d):
                    if root_i != self.find(i+self.strides[d]):
                        region_projection[3*i+1] = 1
                        continue
            region_projection[3*i+2] = self.region_size[root_i]
        return region_projection

    # @cython.boundscheck(False)
    def union(self, int i, int j):
        if (i < 0) or (i > self.n_points) or (j < 0) or (j > self.n_points):
            raise ValueError("Out of bounds index.")
        cdef int root_i, root_j
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j and not (self.seed[root_i] and self.seed[root_j]):
            self._n_sets -= 1
            any_seed = self.seed[root_i] or self.seed[root_j]
            if any_seed:
                self.seed[root_i] = True
                self.seed[root_j] = True
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self.region_size[root_j] += self.region_size[root_i]
                return True
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self.region_size[root_i] += self.region_size[root_j]
                return True
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
                self.region_size[root_j] += self.region_size[root_i]
                return True
        else:
            return False

    cdef int _apply_seeding(self, int i):
        if self._check_seed(i) == 1:
            if self.add_seed(i):
                self.action_counter += 1
                return 1
        return 0

    cdef int _apply_merging(self, int i, int d):
        stride = self.strides[d]
        if self._check_bounds(i, d):
            if self._check_merge(i, i+stride) == 1:
                # stride = self.strides[0]
                # merge index and index + stride, where stride is in direction k
                if self.union(i,i+stride):
                    self.action_counter += 1
                    return 1
        return 0

    cdef i_type _check_seed(self, int i):
        return 1

    cdef i_type _check_merge(self, int i, int j):
        return 1

    def update_MSF(self, int e):
        cdef int i, d
        i = e / (self.n_dims + 1)
        if e % (self.n_dims + 1) == 0:
            return self._apply_seeding(i)
        else:
            # find direction in which to merge
            d = e % (self.n_dims + 1) - 1
            return self._apply_merging(i, d)

    def kruskal_iteration(self, np.ndarray[long, ndim=1] edge_list, int max_iterations):
        cdef int i, e, stride, d, r
        for e in edge_list:
            if max_iterations != 0 and self.action_counter >= max_iterations:
                break
            self.actions[e] = self.update_MSF(e)

    def get_flat_applied_action(self):
        cdef np.ndarray[long, ndim=1] flat_actions
        flat_actions =  np.zeros(self.n_points * (self.n_dims + 1), dtype=long)
        cdef int i
        for i in range(self.n_points*(self.n_dims+1)):
            flat_actions[i] = self.actions[i]
        return flat_actions

    def get_seed_map(self):
        cdef np.ndarray[long, ndim=1] flat_actions
        flat_actions =  np.zeros(self.n_points, dtype=long)
        cdef int i

        for i in range(self.n_points):
            flat_actions[i] = self.actions[3 * i]
        return flat_actions