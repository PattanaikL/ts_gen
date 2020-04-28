from __future__ import print_function
import os, time
import numpy as np
import tensorflow as tf

import GNN


class G2C:
    def __init__(self, num_gpus=1, max_size=29, build_backprop=True,
        node_features=7, edge_features=3, layers=2, hidden_size=128, iterations=3, input_data=None):
        """ Initialize a Graph-to-coordinates network """
        # tf.set_random_seed(42)

        self.dims = {
            "max": max_size,
            "nodes": node_features,
            "edges": edge_features
        }

        self.hyperparams = {
            "node_layers": layers,
            "node_hidden": hidden_size,
            "edge_layers": layers,
            "edge_hidden": hidden_size,
            "iterations": iterations
        }

        with tf.variable_scope("Inputs"):
            if not input_data:
                # Placeholders
                placeholders = {}
                placeholders["nodes"] = tf.placeholder(
                    tf.float32,
                    [None, self.dims["max"], self.dims["nodes"]],
                    name="Nodes"
                )
                placeholders["edges"] = tf.placeholder(
                    tf.float32,
                    [None, self.dims["max"], self.dims["max"], self.dims["edges"]],
                    name="Edges"
                )
                placeholders["sizes"] = tf.placeholder(
                    tf.int32, [None], name="Sizes"
                )
                placeholders["coordinates"] = tf.placeholder(
                    tf.float32,
                    [None, self.dims["max"], 3],
                    name="Coordinates"
                )
                self.placeholders = placeholders
            else:
                self.placeholders = input_data

        # Build graph neural network
        V, E, mask_V, mask_E = GNN.GNN(
            self.placeholders["nodes"],
            self.placeholders["edges"],
            self.placeholders["sizes"],
            iterations=self.hyperparams["iterations"],
            node_layers=self.hyperparams["node_layers"],
            node_hidden=self.hyperparams["node_hidden"],
            edge_layers=self.hyperparams["edge_layers"],
            edge_hidden=self.hyperparams["edge_hidden"]
        )
        self.tensors = {"V": V, "E": E}
        mask_D = tf.squeeze(mask_E, 3)
        self.masks = { "V": mask_V, "E": mask_E, "D": mask_D}
        tf.summary.image("EdgeOut", E[:,:,:,:3])

        # Build featurization
        with tf.variable_scope("DistancePredictions"):
            E = GNN.MLP(
                E, self.hyperparams["edge_layers"],
                self.hyperparams["edge_hidden"]
            )
            self.tensors["embedding"] = tf.reduce_sum(E, axis=[1,2])
            # Make unc. distance and weight predictions
            E_out = tf.layers.dense(E, 2, use_bias=True, name="edge_out")
            # Symmetrize
            E_out = E_out + tf.transpose(E_out, [0,2,1,3])  # permuting on i,j

            # Distance matrix prediction
            D_init = tf.get_variable(
                "D_init", (), initializer=tf.constant_initializer([4.])
            )
            # Enforce positivity
            D = tf.nn.softplus(D_init + E_out[:,:,:,0])
            # Enforce self-distance = 0
            D = self.masks["D"] * tf.linalg.set_diag(
                D, tf.zeros_like(tf.squeeze(self.masks["V"],2))
            )
            self.tensors["D_init"] = D
            tf.summary.image("DistancePred", tf.expand_dims(D,3))

            # Weights prediction
            # W = tf.nn.sigmoid(E_out[:,:,:,1])
            W = tf.nn.softplus(E_out[:,:,:,1])
            self.tensors["W"] = W
            tf.summary.image("W", tf.expand_dims(W,3))

            self.debug_op = tf.add_check_numerics_ops()

        with tf.variable_scope("Reconstruct"):
            # B = self.distance_to_gram(D, self.masks["D"])
            # tf.summary.image("B", tf.expand_dims(B,3))
            # X = self.low_rank_approx(B)
            # X = self.low_rank_approx_weighted(B, W)
            # X = self.low_rank_approx_weighted(B, W)

            # Minimize the objective with unrolled gradient descent
            X = self.dist_nlsq(D, W, self.masks["D"])
            self.tensors["X"] = X

        with tf.variable_scope("Loss"):
            # RMSD Loss
            self.loss, self.tensors["X"] = self.rmsd(X, 
                self.placeholders["coordinates"], self.masks["V"]
            )
            tf.summary.scalar("LossRMSD", self.loss)

            # Difference of distances
            D_model = self.masks["D"] * self.distances(X)
            D_target = self.masks["D"] * self.distances(self.placeholders["coordinates"])

            tf.summary.image("DistModel", tf.expand_dims(D_model,3))
            tf.summary.image("DistTarget", tf.expand_dims(D_target,3))
            tf.summary.histogram("DistModel", D_model)
            tf.summary.histogram("DistTarget", D_target)

            self.loss_distance_all = self.masks["D"] * tf.abs(D_model - D_target)
            self.loss_distance = tf.reduce_sum(self.loss_distance_all) / tf.reduce_sum(self.masks["D"])
            tf.summary.scalar("LossDistance", self.loss_distance)
        # self.debug_op = tf.add_check_numerics_ops()

        with tf.variable_scope("Optimization"):
            opt = tf.train.AdamOptimizer(learning_rate=0.0001)
            gvs = opt.compute_gradients(self.loss_distance)  # gvs is list of (gradient, variable) pairs
            gvs_clipped, global_norm = self.clip_gradients(gvs)
            self.train_op = tf.cond(
                tf.debugging.is_finite(global_norm),        # if global norm is finite
                lambda: opt.apply_gradients(gvs_clipped),   # apply gradients
                lambda: tf.no_op()                          # otherwise do nothing

            )
        # self.debug_op = tf.add_check_numerics_ops()
        return

    def distance_to_gram(self, D, mask):
        """ Convert distance to gram matrix """
        N_f32 = tf.to_float(self.placeholders["sizes"])
        D = tf.square(D)
        D_row = tf.reduce_sum(D, 1, keep_dims=True) \
            / tf.reshape(N_f32, [-1,1,1])
        D_col = tf.reduce_sum(D, 2, keep_dims=True) \
            / tf.reshape(N_f32, [-1,1,1])
        D_mean = tf.reduce_sum(D, [1,2], keep_dims=True) \
            / tf.reshape(tf.square(N_f32), [-1,1,1])
        B = mask * -0.5 * (D - D_row - D_col + D_mean)
        return B

    def low_rank_approx(self, A, k=3):
        with tf.variable_scope("LowRank"):
            A = self.dither(A)
            # Recover X
            S, U, V = tf.svd(
                A, full_matrices=True, compute_uv=True, name="SVD"
            )
            X = tf.sqrt(tf.expand_dims(S[:,:k],1) + 1E-3) * U[:,:,:k]
            # Debug SV collisions
            # tf.summary.image("A", tf.expand_dims(A,3))
            # S_gap = tf.reduce_min(tf.abs(S[:,1:] - S[:,:-1]))
            # tf.summary.scalar("S_gap", S_gap)
        return X

    def low_rank_approx_power(self, A, k=3, num_steps=10):
        # Low rank approximation
        with tf.variable_scope("LowRank"):
            A_lr = A
            u_set = []
            for kx in range(k):
                # Initialize Eigenvector
                u = tf.expand_dims(tf.random_normal(tf.shape(A)[:-1]),-1)
                # Power iteration
                for j in range(num_steps):
                    u = tf.nn.l2_normalize(u, 1, epsilon=1e-3)
                    u = tf.matmul(A_lr, u)
                # Rescale by sqrt(eigenvalue)
                eig_sq = tf.reduce_sum(tf.square(u), 1, keep_dims=True)
                u = u / tf.pow(eig_sq + 1E-2, 0.25)
                u_set.append(u)
                A_lr = A_lr - tf.matmul(u, u, transpose_b=True)
            X = tf.concat(axis=2, values=u_set)
        return X

    def low_rank_approx_weighted(self, A, W, k=3, num_iter=10):
        with tf.variable_scope("WeightedLowRank"):
            WA = W * A
            W_comp = (1. - W)
            A_i = tf.zeros_like(A)
            for i in range(num_iter):
                # Low-rank approximation
                # X = self.low_rank_approx_power(A, k=3)
                k_i = max(3 + (self.dims["max"] - 3) * (num_iter - i-1) / num_iter, 3)
                print(k_i)
                X = self.low_rank_approx(A, k=k_i)
                # Rebuild matrix
                if i < num_iter - 1:
                    X = X + tf.random_normal(tf.shape(X)) / (i + 1)
                    A_i = tf.matmul(X, X, transpose_b = True)
        return X

    def dither(self, A, symmetrize=True, radius=1E-2):
        with tf.variable_scope("Dither"):
            A = A * self.masks["D"]
            A = A + 1E-2 * tf.random_normal(tf.shape(A))
            perturb = tf.random_shuffle(tf.range(tf.to_float(tf.shape(A)[1])))
            A = A + 1E-2 * tf.expand_dims(tf.matrix_diag(perturb), 0)
            if symmetrize:
                A = 0.5 * (A + tf.transpose(A,[0,2,1]))
        return A

    def dist_nlsq(self, D, W, mask):
        """ Solve a nonlinear distance geometry problem by nonlinear least
            squares

            Objective is Sum_ij w_ij (D_ij - |x_i - x_j|)^2
        """
        T = 100
        # eps = tf.exp(tf.get_variable(
        #     "eps", (), initializer=tf.constant_initializer([np.log(0.1)])
        # ))
        # tf.summary.scalar("eps", eps)
        # # Max speed
        # alpha = tf.exp(tf.get_variable(
        #     "alpha", (),
        #     initializer=tf.constant_initializer([np.log(5.0)])
        # ))
        # tf.summary.scalar("alpha", alpha)
        eps = 0.1
        alpha = 5.0
        alpha_base = 0.1

        def gradfun(X):
            """ Grad function """
            D_X = self.distances(X)
            # Energy calculation
            U = tf.reduce_sum(mask * W * tf.square(D - D_X), [1,2]) \
                / tf.reduce_sum(mask, [1,2])
            U = tf.reduce_sum(U)
            # Gradient calculation
            g = tf.gradients(U, X)[0]

            # DEBUG: 
            g = tf.cond(
                tf.debugging.is_finite(tf.reduce_sum(g)),
                lambda: g,
                lambda: tf.Print(
                    g, [tf.reduce_sum(tf.square(X)), tf.reduce_sum(tf.square(g)), tf.reduce_sum(D), tf.reduce_sum(U)], 
                    message="Error with Gradient"
                )
            )
            return g

        def stepfun(t, x_t):
            """ Step function """
            g = gradfun(x_t)
            dx = -eps * g

            # Speed clipping (How fast in Angstroms)
            speed = tf.sqrt(
                tf.reduce_sum(tf.square(dx), 2, keep_dims=True) + 1E-3
            )
            # Alpha sets max speed (soft trust region)
            alpha_t = alpha_base + (alpha - alpha_base) * tf.to_float((T - t) / T)
            scale = alpha_t * tf.tanh(speed / alpha_t) / speed
            dx *= scale

            x_new = x_t + dx

            # DEBUG: 
            x_new = tf.cond(
                tf.debugging.is_finite(tf.reduce_sum(x_new)),
                lambda: x_new,
                lambda: tf.Print(
                    x_new, [t, tf.reduce_sum(x_new), tf.reduce_sum(g)], 
                    message="Error with Coordinates"
                )
            )
            return t + 1, x_new

        # Initialization
        B = self.distance_to_gram(D, mask)
        x_init = self.low_rank_approx_power(B)

        # Prepare simulation
        x_init += tf.random_normal([tf.shape(D)[0], self.dims["max"], 3])
        state_init = [0, x_init]

        # Optimization loop
        _, x_final = tf.while_loop(
            lambda t, x: t < T, stepfun, state_init, swap_memory=False
        )
        return x_final

    def rmsd(self, X1, X2, mask_V):
        """ RMSD between two structures """
        with tf.variable_scope("RMSD"):
            X1 = X1 - tf.reduce_sum(mask_V * X1,axis=1,keep_dims=True) \
                / tf.reduce_sum(mask_V, axis=1,keep_dims=True)
            X2 = X2 - tf.reduce_sum(mask_V * X2,axis=1,keep_dims=True) \
                / tf.reduce_sum(mask_V, axis=1,keep_dims=True)
            X1 *= mask_V
            X2 *= mask_V
            eps = 1E-2
            X1_perturb = X1 + eps * tf.random_normal(tf.shape(X1))
            X2_perturb = X2 + eps * tf.random_normal(tf.shape(X1))
            A = tf.matmul(X2_perturb, X1_perturb, transpose_a=True)
            S,U,V = tf.svd(A, full_matrices=True, compute_uv=True, name="SVD")

            X1_align = tf.matmul(U, tf.matmul(V, X1, transpose_b = True))
            X1_align = tf.transpose(X1_align, [0,2,1])
            MSD = tf.reduce_sum(mask_V * tf.square(X1_align - X2), [1,2]) \
                / tf.reduce_sum(mask_V, [1, 2])
            RMSD = tf.reduce_mean(tf.sqrt(MSD + 1E-3))
        return RMSD,  X1_align

    def clip_gradients(self, gvs):
        """ Clip the gradients """
        with tf.name_scope("Clip"):
            grads, gvars = list(zip(*gvs)[0]), list(zip(*gvs)[1])
            clipped_grads, global_norm = tf.clip_by_global_norm(grads, 10)
            tf.summary.scalar('gradient_norm', global_norm)
            clipped_gvs = zip(clipped_grads, gvars)
        return clipped_gvs, global_norm

    def distances(self, X):
        """ Compute Euclidean distances from X """
        with tf.variable_scope("SquaredDistances"):
            Dsq = tf.square(tf.expand_dims(X, 1) - tf.expand_dims(X, 2))
            D = tf.sqrt(tf.reduce_sum(Dsq, 3) + 1E-2)
        return D
