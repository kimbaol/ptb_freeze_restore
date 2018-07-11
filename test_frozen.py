import tensorflow as tf
import numpy as np
import reader
import time

frozen_graph='frozen_test.pb'

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

graph = load_graph(frozen_graph)

output_op = graph.get_operation_by_name('import/Test/Model/Sum')
input_op_x = graph.get_operation_by_name('import/Test/TestInput/StridedSlice')
input_op_y = graph.get_operation_by_name('import/Test/TestInput/StridedSlice_1')
## hidden state input
init_state_op0 = graph.get_operation_by_name('import/Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros') ##c(t-1)
init_state_op1 = graph.get_operation_by_name('import/Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1') ##h(t-1)
init_state_op2 = graph.get_operation_by_name('import/Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros') ##c(t-1)
init_state_op3 = graph.get_operation_by_name('import/Test/Model/MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1')##h(t-1)
## hidden state output
state_out0 = graph.get_operation_by_name('import/Test/Model/RNN/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1') ##c(t)
state_out1 = graph.get_operation_by_name('import/Test/Model/RNN/RNN/multi_rnn_cell/cell_0/basic_lstm_cell/Mul_2') ##h(t)
state_out2 = graph.get_operation_by_name('import/Test/Model/RNN/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/Add_1')  ##c(t)
state_out3 = graph.get_operation_by_name('import/Test/Model/RNN/RNN/multi_rnn_cell/cell_1/basic_lstm_cell/Mul_2')  ##h(t)


data_path = 'data/data/'
batch_size = 1
num_steps = 1
name = 'TestInput'
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, _ = raw_data
data_x, data_y = reader.ptb_producer(test_data, batch_size, num_steps, name=name)
epoch_size = ((len(test_data) // batch_size) - 1) // num_steps

xval, yval = [], []
with tf.Session() as session:
   coord = tf.train.Coordinator()
   tf.train.start_queue_runners(session, coord=coord)
   try:
     for i in range(epoch_size):
         x, y = session.run([data_x, data_y])
         xval.append(x)
         yval.append(y)
   finally:
     coord.request_stop()
     coord.join()

start = time.time()
with tf.Session(graph=graph) as session:
    costs = 0.0
    feed_dict = {}
    state0 = state1 = state2 = state3 = []
    for i in range(epoch_size): 
        feed_dict = {input_op_x.outputs[0]:xval[i], input_op_y.outputs[0]:yval[i]}
        if i>0 :
            feed_dict.update({init_state_op0.outputs[0]: state0, init_state_op1.outputs[0]: state1,
                              init_state_op2.outputs[0]: state2, init_state_op3.outputs[0]: state3,})
        loss, state0, state1, state2, state3 \
            = session.run([output_op.outputs[0], 
                          state_out0.outputs[0],
                          state_out1.outputs[0],
                          state_out2.outputs[0],
                          state_out3.outputs[0]],
                          feed_dict=feed_dict)
        costs += loss
end = time.time()
print("time used: " + str(end-start))
#print(costs)
#print(epoch_size)
print('perplexity: ' + str(np.exp(costs / epoch_size)))


