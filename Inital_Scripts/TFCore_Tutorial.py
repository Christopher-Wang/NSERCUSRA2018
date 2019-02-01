"""
Tensorflow implementation of the Conv2D_LSTM arcitecture. 
"""
import numpy as np
import tensorflow as tf




y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)

#Define the data, input and label
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

#Create a dense layer and predict
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

#Instantiate the session
sess = tf.Session()

#Define the initializer and initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

#Predict to show values
print(sess.run(y_pred))

#Define the loss function and calculate th first value of the loss function
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

#Define an optimizer and 
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Train
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

#Predict to show values
print(sess.run(y_pred))
