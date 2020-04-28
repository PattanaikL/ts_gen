from __future__ import print_function
import os, sys, time, random
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

sys.path.insert(0, "model/")
from util import render_pymol
from G2C import G2C

# rdkit
from rdkit import Chem, Geometry

from optparse import OptionParser


parser = OptionParser()
parser.add_option("-r", "--restore", dest="restore", default=None)
parser.add_option("-l", "--layers", dest="layers", default=2)
parser.add_option("-s", "--hidden_size", dest="hidden_size", default=128)
parser.add_option("-i", "--iterations", dest="iterations", default=3)
parser.add_option("-g", "--gpu", dest="gpu", default=0)

opts, args = parser.parse_args()
layers = int(opts.layers)
hidden_size = int(opts.hidden_size)
iterations = int(opts.iterations)
gpu = str(opts.gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

reactantFile = 'data/intra_rxns_reactants.sdf'
tsFile = 'data/intra_rxns_ts.sdf'
productFile = 'data/intra_rxns_products.sdf'

QUEUE = True
BATCH_SIZE = 8
EPOCHS = 200
best_val_loss = 9e99

# Load dataset
print("Loading datset")
start = time.time()
data = [Chem.SDMolSupplier(reactantFile, removeHs=False, sanitize=False),
        Chem.SDMolSupplier(tsFile, removeHs=False, sanitize=False),
        Chem.SDMolSupplier(productFile, removeHs=False, sanitize=False)]
data = [(x,y,z) for (x,y,z) in zip(data[0],data[1],data[2]) if (x,y,z)]

elapsed = time.time() - start
print(" ... loaded {} molecules in {:.2f}s".format(len(data), elapsed))

# Dataset specific dimensions
elements = "HCNO"
num_elements = len(elements)
max_size = max([x.GetNumAtoms() for x,y,z in data])
print(max_size)

# Splitting
#np.random.seed(42)
#random.seed(42)
N_data = len(data)
idx = np.arange(N_data)
np.random.shuffle(idx)
idx = idx.tolist()
N_test = int(round(N_data / 10))

N_valid = N_test
N_train = N_data - N_valid - N_test

data_test = [data[i] for i in idx[:N_test]]
data_valid = [data[i] for i in idx[N_test:N_test + N_valid]]
data_train = [data[i] for i in idx[N_test + N_valid:]]

# Build basepath for experiment
base_folder = time.strftime("log/%y%b%d_%I%M%p/", time.localtime())
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
# base_folder = 'log/{}layers_{}hs_{}its/'.format(layers, hidden_size, iterations)
# if not os.path.exists(base_folder):
#     os.makedirs(base_folder)

train_ts_file = base_folder + 'train_ts.sdf'
train_reactant_file = base_folder + 'train_reactants.sdf'
train_products_file = base_folder + 'train_products.sdf'

train_ts_writer = Chem.SDWriter(train_ts_file)
train_reactant_writer = Chem.SDWriter(train_reactant_file)
train_product_writer = Chem.SDWriter(train_products_file)

for i in range(len(data_train)):
    train_ts_writer.write(data_train[i][0])
    train_reactant_writer.write(data_train[i][1])
    train_product_writer.write(data_train[i][2])


def prepare_batch(batch_mols):

    # Initialization
    size = len(batch_mols)
    V = np.zeros((size, max_size, num_elements + 1), dtype=np.float32)
    E = np.zeros((size, max_size, max_size, 3), dtype=np.float32)
    sizes = np.zeros(size, dtype=np.int32)
    coordinates = np.zeros((size, max_size, 3), dtype=np.float32)

    # Build atom features
    for bx in range(size):
        reactant, ts, product = batch_mols[bx]
        N_atoms = reactant.GetNumAtoms()
        sizes[bx] = int(N_atoms)

        # Topological distances matrix
        MAX_D = 10.
        D = (Chem.GetDistanceMatrix(reactant) + Chem.GetDistanceMatrix(product)) / 2
        D[D > MAX_D] = 10.

        D_3D_rbf = np.exp(-((Chem.Get3DDistanceMatrix(reactant) + Chem.Get3DDistanceMatrix(product)) / 2))  # squared

        for i in range(N_atoms):
            # Edge features
            for j in range(N_atoms):
                E[bx, i, j, 2] = D_3D_rbf[i][j]
                if D[i][j] == 1.:  # if stays bonded
                    if reactant.GetBondBetweenAtoms(i, j).GetIsAromatic():
                        E[bx, i, j, 0] = 1.
                    E[bx, i, j, 1] = 1.

            # Recover coordinates; adapted for all
            # for k, mol_typ in enumerate([reactant, ts, product]):
            pos = ts.GetConformer().GetAtomPosition(i)
            np.asarray([pos.x, pos.y, pos.z])
            coordinates[bx, i, :] = np.asarray([pos.x, pos.y, pos.z])

            # Node features
            atom = reactant.GetAtomWithIdx(i)
            e_ix = elements.index(atom.GetSymbol())
            V[bx, i, e_ix] = 1.
            V[bx, i, num_elements] = atom.GetAtomicNum() / 10.
            # V[bx, i, num_elements + 1] = atom.GetExplicitValence() / 10.

    # print(np.sum(np.square(V)),np.sum(np.square(E)), sizes)
    batch_dict = {
        "nodes": tf.constant(V),
        "edges": tf.constant(E),
        "sizes": tf.constant(sizes),
        "coordinates": tf.constant(coordinates)
    }
    return batch_dict


##########################################################################################
with tf.variable_scope("Dataset"):

    dtypes = [tf.float32, tf.float32, tf.int32, tf.float32]
    names = ['nodes', 'edges', 'sizes', 'coordinates']
    shapes = [[BATCH_SIZE, max_size, num_elements + 1], [BATCH_SIZE, max_size, max_size, 3], [BATCH_SIZE],
              [BATCH_SIZE, max_size, 3]]
    number_of_threads_train_n = 3

    ds_train = tf.data.Dataset.from_tensor_slices(prepare_batch(data_train)).cache().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds_valid = tf.data.Dataset.from_tensor_slices(prepare_batch(data_valid)).cache().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(ds_train)
    validation_init_op = iterator.make_initializer(ds_valid)

##########################################################################################


# Build model
print("Building model")
start = time.time()
dgnn = G2C(
    max_size=max_size, node_features=num_elements + 1, edge_features=3,
    layers=layers, hidden_size=hidden_size, iterations=iterations, input_data=next_element
)
elapsed = time.time() - start
print(" ... model built in {:.2f}s".format(elapsed))

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print('Total trainable variables: {}'.format(total_parameters))

# Launch session
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # Initialization
    print("Initializing")
    start = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)
    elapsed = time.time() - start
    print(" ...init in " + str(elapsed) + "s\n")

    # Build batch summaries and TensorBoard writer
    summary_op = tf.summary.merge_all()
    print("Setting up saver")
    start = time.time()
    summary_writer = tf.summary.FileWriter(base_folder, sess.graph)

    # Variable saving
    saver = tf.train.Saver()
    if opts.restore is not None:
        saver.restore(sess, opts.restore)
    elapsed = time.time() - start
    print(" set up in " + str(elapsed) + " s\n")

    counter = 0

    for epoch in range(EPOCHS):

        sess.run(training_init_op)
        batches_trained = 0
        epoch_start = time.time()

        try:
            while True:
                batch_start = time.time()

                _, _, summ = sess.run(
                    [dgnn.train_op, dgnn.debug_op, summary_op])

                summary_writer.add_summary(summ, counter)
                batches_trained += 1
		counter += 1

                print('Training: ', batches_trained, time.time()-batch_start)
                # if batches_trained % 10 == 0:
                #     print(batches_trained)

        except tf.errors.OutOfRangeError as e:
            pass
        sess.run(validation_init_op)

        X = np.empty([len(data_valid), max_size, 3])
        valid_loss_all = np.empty([len(data_valid), max_size, max_size])
        D_mask = np.empty([len(data_valid), max_size, max_size])
        batches_validated = 0

        try:

            while True:
                batch_start = time.time()

                _, X[batches_validated * BATCH_SIZE:(batches_validated + 1) * BATCH_SIZE, :, :], \
                valid_loss_all[batches_validated * BATCH_SIZE:(batches_validated + 1) * BATCH_SIZE, :, :], \
                D_mask[batches_validated * BATCH_SIZE:(batches_validated + 1) * BATCH_SIZE, :, :] = sess.run(
                    [dgnn.debug_op, dgnn.tensors["X"], dgnn.loss_distance_all, dgnn.masks["D"]])

                batches_validated += 1
                print('Validation: ', batches_validated, time.time()-batch_start)

        except tf.errors.OutOfRangeError as e:
            pass

        other_calcs_start = time.time()
        losses = [np.sum(valid_loss_all[i] * D_mask[i]) / np.sum(D_mask[i]) for i in range(X.shape[0])]
        val_loss = np.mean(np.asarray(losses))

        # Make copy of molecule (early checks)
        if epoch < 5:
            bx = 0
            mol_target = data_valid[bx][1]  # transition state geom
            mol = Chem.Mol(mol_target)
            for i in range(mol.GetNumAtoms()):
                x = X[bx, i, :].tolist()
                mol.GetConformer().SetAtomPosition(
                    i, Geometry.Point3D(x[0], x[1], x[2])
                )

            render_pymol(mol, base_folder + "step{}_model.png".format(epoch), width=600, height=400)
            render_pymol(mol_target, base_folder + "step{}_target.png".format(epoch), width=600, height=400)

        # save_path = saver.save(sess, base_folder + "step{}_model.ckpt".format(epoch))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = saver.save(sess, base_folder + "best_model.ckpt")
        save_path_last = saver.save(sess, base_folder + "last_model.ckpt")
        # print('Other Calcs: ', time.time()-other_calcs_start)
        print("Validation Loss: {}".format(val_loss))
        # print("Time to train and validate epoch: {} s".format(time.time()-epoch_start))


print("Best Validation Loss: {}".format(best_val_loss))
print("Hyperparameters:")
print("Batch size: {}".format(BATCH_SIZE))
print("Layers: {}".format(layers))
print("Hidden size: {}".format(hidden_size))
print("Iterations: {}".format(iterations))

# Save test data
test_ts_file = base_folder + 'test_ts.sdf'
test_reactant_file = base_folder + 'test_reactants.sdf'
test_products_file = base_folder + 'test_products.sdf'

test_ts_writer = Chem.SDWriter(test_ts_file)
test_reactant_writer = Chem.SDWriter(test_reactant_file)
test_product_writer = Chem.SDWriter(test_products_file)

for i in range(len(data_test)):
    test_reactant_writer.write(data_test[i][0])
    test_ts_writer.write(data_test[i][1])
    test_product_writer.write(data_test[i][2])
