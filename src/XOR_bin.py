from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

#what does the same code look like with Tensor Board
import time
sess = tf.Session()
log_dir = "./tesnorBoardFiles/NSC211_BKlecture_code" #from currdir of notebook
start_time = time.time() #record start time later 

#input data
XOR_X = [[0,0],[0,1],[1,0],[1,1]] #input
XOR_Y = [[0],[1],[1],[0]] #predicted

#placeholders, now we need one for predicted vals too
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input") 
#constrain rand. values
w1 = tf.Variable(tf.random_uniform([2,2], .7, 1.3), tf.float32) #W
#tf.summary.scalar('mean', mean)   
w2 = tf.Variable(tf.random_uniform([2,1], -2, 1), tf.float32) #w
b1 = tf.Variable(tf.zeros([2]), tf.float32) #c
b2 = tf.Variable(tf.zeros([1]), tf.float32) #b

#operation nodes
transformedH = tf.nn.relu(tf.matmul(x_,w1) + b1) #hidden layer with rect. linear act. func.
linear_model = tf.matmul(transformedH, w2) + b2
summary = tf.summary.scalar("predicted", linear_model)


#MSE
loss = tf.reduce_sum(tf.square(linear_model - y_)) #create error vector.We call tf.square to square that error.

#gradient descent 
optimizer = tf.train.GradientDescentOptimizer(0.01) #0.01 is learning rate
train = optimizer.minimize(loss) #feed optimizer loss function 
summary = tf.summary.scalar("loss", curr_loss)




init = tf.global_variables_initializer()

sess.run(init)
# Build the summary Tensor based on the TF collection of Summaries.
#summary = tf.summary.merge_all()
# Instantiate a SummaryWriter to output summaries and the Graph. after sess created?
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)


#train it
for i in range(10000):
        feed_dict = {x_: XOR_X, y_: XOR_Y}
        sess.run(train, feed_dict)
        curr_loss = sess.run(loss, feed_dict)
        curr_predict = sess.run(loss, feed_dict)
        #currsumm = sess.run(summary,feed_dict)
        #print("loss:%s\n"%(curr_loss))
        # Write the summaries and print an overview fairly often.
        duration = time.time() - start_time
        if i % 100 == 0:
            # Print status to stdout.
            #loss_array = curr_loss.eval(session=sess.run)
            print("Step %d: loss = %.2f (%.3f sec)" % (i, curr_loss, duration))
            # Update the events file.
            summary_str = sess.run(summary) #, feed_dict
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()


#stake a look at the results
predictions = sess.run(linear_model, {x_: XOR_X}) 
curr_w1, curr_w2, curr_b1, curr_b2, curr_loss  = sess.run([w1, w2, b1, b2, loss], {x_: XOR_X, y_: XOR_Y})
hidlay  = sess.run(transformedH, {x_: XOR_X, y_: XOR_Y})
print("predictions:\n %s\n hlayout:\n %s\n"%(predictions,hidlay))
print("w1:\n %s \nw2:\n %s \nb1: %s \nb2: %s \nloss: %s"%(curr_w1, curr_w2, curr_b1, curr_b2, curr_loss))