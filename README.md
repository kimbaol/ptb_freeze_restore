# ptb_freeze_restore
Freeze the ptb model from tensorflow Tutorial then load the frozen model 

## steps:
1. clone ptb model from tensorflow/models/tutorials/rnn/ptb/ then train it.
2. freeze_graph --input_graph=./model/graph.pbtxt --input_checkpoint=(ckpt file prefix) --output_graph=frozen_test.pb --output_node_names=Test/Model/Sum
3. python test_frozen.py

