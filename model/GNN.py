from __future__ import print_function
import tensorflow as tf
import numpy as np

def GNN(V_init, E_init, sizes, iterations=3, edge_layers = 2,
    edge_hidden = 100, node_layers = 2, node_hidden = 100, act=tf.nn.relu):
    """ Graph neural network with node & edge updates """
    V, E = V_init, E_init

    # Get dimensions
    N_v = int(V.get_shape()[1])
    C_v = int(V.get_shape()[2])
    C_e = int(E.get_shape()[3])

    with tf.variable_scope("GraphNeuralNet"):
        with tf.variable_scope("Masks"):
            mask = tf.sequence_mask(
                sizes, maxlen=N_v, dtype=tf.float32, name="Mask1D"
            )
            mask_V = tf.expand_dims(mask, 2)
            mask_E = tf.expand_dims(mask_V,1) * tf.expand_dims(mask_V,2)
        
        # Initialize hidden state
        with tf.variable_scope("NodeInit"):
            V = mask_V * MLP(V, node_layers, node_hidden)
        with tf.variable_scope("EdgeInit"):
            E = mask_E * MLP(E, edge_layers, edge_hidden)
            tf.summary.image("Edge", E[:,:,:,:3])

        for i in range(iterations):
            # with tf.variable_scope("Iteration{}".format(i)):
            #     reuse = None
            with tf.name_scope("Iteration{}".format(i)):
                reuse = True if i > 0 else None
                with tf.variable_scope("EdgeUpdate", reuse=reuse):
                    # Update edges given {V,E}
                    f = PairFeatures(
                        V, E, edge_hidden, reuse=reuse, name="EdgeFeatures", activation=act
                    )
                    dE = MLP(
                        f, edge_layers, edge_hidden, name="EdgeMLP", activation=act, reuse=reuse  # changed
                    )
                    # dE = tf.layers.dropout(dE, dropout, training=bool(dropout))
                    E = E + mask_E * dE
                with tf.variable_scope("NodeUpdate", reuse=reuse):
                    # Update nodes given {V,E'}
                    # f = PairFeatures(
                    #     V, E, node_hidden, reuse=reuse, name="NodeFeatures", activation=act
                    # )
                    tf.summary.image("EdgeOut", E[:,:,:,:3])
                    dV = MLP(
                        E, node_layers, node_hidden, name = "NodeMessages", activation=act, reuse=reuse
                    )
                    dV = tf.reduce_sum(dV, 2)
                    dV = MLP(
                        dV, node_layers, node_hidden, name = "NodeMLP", activation=act, reuse=reuse  # changed
                    )
                    # dV = tf.layers.dropout(dV, dropout, training=bool(dropout))
                    V = V + mask_V * dV
    return V, E, mask_V, mask_E

def PairFeatures(V, E, out_dim, reuse=None, name="PairFeatures", activation=tf.nn.relu):
    """ Build pair features from V,E """

    with tf.variable_scope(name, reuse=reuse):
        f_ij = tf.layers.dense(
            E, out_dim, use_bias=True, name="f_ij", reuse=reuse
        )
        f_i = tf.expand_dims(tf.layers.dense(
            V, out_dim, use_bias=False, name="f_i", reuse=reuse
        ), 1)
        f_j = tf.expand_dims(tf.layers.dense(
            V, out_dim, use_bias=False, name="f_j", reuse=reuse
        ), 2)
        f = activation(f_ij + f_i + f_j)
    return f

def MLP(input_h, num_layers, out_dim, name="MLP", activation=tf.nn.relu, reuse=None):
    """ Build a multilayer perceptron """
    with tf.variable_scope(name, reuse=reuse):
        h = input_h
        for i in range(num_layers):
            h = tf.layers.dense(
                h, out_dim, 
                use_bias=True, activation=activation,
                name="mlp{}".format(i),
                reuse = reuse
            )
        h = tf.layers.dense(
            h, out_dim, use_bias=True, name="mlp_out", reuse=reuse
        )
    return h