import numpy as np
import theano
from theano import tensor

from utilities import create_streams
from largesparse_theano_op.op import LargeSparseTargets

theano.config.floatX = 'float32'
floatX = theano.config.floatX


##########
# CONFIG #
##########
D = 793471  # vocabulary size
m = 100  # batch size
n_grams = 6
embedding_size = 100
init_var = 0.01
invup_mode = 1
hasOutputBias = True  # Presence of a bias term for the output layer
learning_rate = 0.1
freq_stabilize = 10


################
# ARCHITECTURE #
################
x = tensor.imatrix('ngrams')

# Embedding layer
embedding_matrix = theano.shared(
    np.random.normal(0, 0.001, (D, embedding_size)).astype(floatX))
buff = tensor.reshape(x, (-1,))
sub = embedding_matrix[buff]
embeddings = tensor.reshape(sub, (x.shape[0], x.shape[1], -1))

# bag of words
bag_of_words = embeddings.mean(axis=1)

# Output layer
if hasOutputBias:
    bag_of_words = tensor.concatenate(
        [bag_of_words,
         tensor.constant(0.001 * np.ones((m, 1), dtype=floatX))], axis=1)
    d = embedding_size + 1
else:
    d = embedding_size

V_mat = np.random.normal(0, 0.001, (D, d)).astype(floatX)
U_mat = np.eye(d, dtype=floatX)
UinvT_mat = np.eye(d, dtype=floatX)
Q_mat = np.dot(V_mat.T, V_mat)

U = theano.shared(U_mat, name="U")
V = theano.shared(V_mat, name="V")
UinvT = theano.shared(UinvT_mat)
Q = theano.shared(Q_mat)
Y_values = theano.shared(np.ones((m, 1),
                                 dtype=theano.config.floatX))

y_indexes = tensor.imatrix('targets')
mse_loss = LargeSparseTargets()(
    V, U, UinvT, Q, bag_of_words, y_indexes, Y_values, learning_rate,
    use_qtilde=0, use_lower=1, invup_mode=invup_mode,
    stabilize_period=freq_stabilize, unfactorize_period=0, debug_print=0)

output = tensor.dot(bag_of_words, tensor.dot(U, V.T))


######################
# TRAINING FUNCTIONS #
######################
print 'Compile functions ...'
# The gradient and updates of the output layer do not need to be specified as
# they are already taken into account by the op LargeSparseTargets
grad = theano.grad(mse_loss, sub)
update = embedding_matrix, tensor.inc_subtensor(sub, -grad * learning_rate)
fun_train = theano.function([x, y_indexes], mse_loss, updates=[update])


###########
# DATASET #
###########
train_stream, valid_stream, vocab = create_streams(m, n_grams)


############
# TRAINING #
############
it = 0
print 'Start training'
for epoch in range(10):
    for x_mat, y_mat in train_stream.get_epoch_iterator():
        cost = fun_train(x_mat, y_mat)
        print 'batch number {}:\t {}'.format(it, cost)
        it += 1
