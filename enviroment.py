import tensorflow as tf

class AgentVRP():

    VEHICLE_CAPACITY = 1.0

    def __init__(self, input):

        depot = input[0]
        loc = input[1]

        self.batch_size, self.n_loc, _ = loc.shape  # (batch_size, n_nodes, 2)

        # Coordinates of depot + other nodes
        self.coords = tf.concat((depot[:, None, :], loc), -2)
        self.demand = tf.cast(input[2], tf.float32)

        # Indices of graphs in batch
        self.ids = tf.range(self.batch_size, dtype=tf.int64)[:, None]

        # State
        self.prev_a = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.from_depot = self.prev_a == 0
        self.used_capacity = tf.zeros((self.batch_size, 1), dtype=tf.float32)

        # Nodes that have been visited will be marked with 1
        self.visited = tf.zeros((self.batch_size, 1, self.n_loc + 1), dtype=tf.uint8)

        # Step counter
        self.i = tf.zeros(1, dtype=tf.int64)

        # Constant tensors for scatter update (in step method)
        self.step_updates = tf.ones((self.batch_size, 1), dtype=tf.uint8)  # (batch_size, 1)
        self.scatter_zeros = tf.zeros((self.batch_size, 1), dtype=tf.int64)  # (batch_size, 1)

    @staticmethod
    def outer_pr(a, b):
        """Outer product of matrices
        """
        return tf.einsum('ki,kj->kij', a, b)

    def get_att_mask(self):
        """ Mask (batch_size, n_nodes, n_nodes) for attention encoder.
            We mask already visited nodes except depot
        """

        # We dont want to mask depot
        att_mask = tf.squeeze(tf.cast(self.visited, tf.float32), axis=-2)[:, 1:]  # [batch_size, 1, n_nodes] --> [batch_size, n_nodes-1]
        
        # Number of nodes in new instance after masking
        cur_num_nodes = self.n_loc + 1 - tf.reshape(tf.reduce_sum(att_mask, -1), (-1,1))  # [batch_size, 1]
        
        att_mask = tf.concat((tf.zeros(shape=(att_mask.shape[0],1),dtype=tf.float32),att_mask), axis=-1)

        ones_mask = tf.ones_like(att_mask)

        # Create square attention mask from row-like mask
        att_mask = AgentVRP.outer_pr(att_mask, ones_mask) \
                            + AgentVRP.outer_pr(ones_mask, att_mask)\
                            - AgentVRP.outer_pr(att_mask, att_mask)
        
        return tf.cast(att_mask, dtype=tf.bool), cur_num_nodes

    def all_finished(self):
        """Checks if all games are finished
        """
        return tf.reduce_all(tf.cast(self.visited, tf.bool))

    def partial_finished(self):
        """Checks if partial solution for all graphs has been built, i.e. all agents came back to depot
        """
        return tf.reduce_all(self.from_depot) and self.i != 0

    def get_mask(self):
        """ Returns a mask (batch_size, 1, n_nodes) with available actions.
            Impossible nodes are masked.
        """

        # Exclude depot
        visited_loc = self.visited[:, :, 1:]

        # Mark nodes which exceed vehicle capacity
        exceeds_cap = self.demand + self.used_capacity > self.VEHICLE_CAPACITY

        # We mask nodes that are already visited or have too much demand
        # Also for dynamical model we stop agent at depot when it arrives there (for partial solution)
        mask_loc = tf.cast(visited_loc, tf.bool) | exceeds_cap[:, None, :] | ((self.i > 0) & self.from_depot[:, None, :])

        # We can choose depot if 1) we are not in depot OR 2) all nodes are visited
        mask_depot = self.from_depot & (tf.reduce_sum(tf.cast(mask_loc == False, tf.int32), axis=-1) > 0)

        return tf.concat([mask_depot[:, :, None], mask_loc], axis=-1)

    def step(self, action):

        # Update current state
        selected = action[:, None]

        self.prev_a = selected
        self.from_depot = self.prev_a == 0

        # We have to shift indices by 1 since demand doesn't include depot
        # 0-index in demand corresponds to the FIRST node
        selected_demand = tf.gather_nd(self.demand,
                                       tf.concat([self.ids, tf.clip_by_value(self.prev_a - 1, 0, self.n_loc - 1)], axis=1)
                                       )[:, None]  # (batch_size, 1)

        # We add current node capacity to used capacity and set it to zero if we return to the depot
        self.used_capacity = (self.used_capacity + selected_demand) * (1.0 - tf.cast(self.from_depot, tf.float32))

        # Update visited nodes (set 1 to visited nodes)
        idx = tf.cast(tf.concat((self.ids, self.scatter_zeros, self.prev_a), axis=-1), tf.int32)[:, None, :]  # (batch_size, 1, 3)
        self.visited = tf.tensor_scatter_nd_update(self.visited, idx, self.step_updates)  # (batch_size, 1, n_nodes)

        self.i = self.i + 1

    @staticmethod
    def get_costs(dataset, pi):

        # Place nodes with coordinates in order of decoder tour
        loc_with_depot = tf.concat([dataset[0][:, None, :], dataset[1]], axis=1)  # (batch_size, n_nodes, 2)
        d = tf.gather(loc_with_depot, tf.cast(pi, tf.int32), batch_dims=1)

        # Calculation of total distance
        # Note: first element of pi is not depot, but the first selected node in the path
        return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
                + tf.norm(d[:, 0] - dataset[0], ord=2, axis=1) # Distance from depot to first selected node
                + tf.norm(d[:, -1] - dataset[0], ord=2, axis=1))  # Distance from last selected node (!=0 for graph with longest path) to depot
