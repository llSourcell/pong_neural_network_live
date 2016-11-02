import tensorflow as tf
import cv2
import pong
import numpy as np
import random
from collections import deque

ACTIONS = 3
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 500000 # frames over which to anneal epsilon
OBSERVE = 50000
REPLAY_MEMORY = 500000
BATCH = 100

def createGraph():
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    #input
    s = tf.placeholder("float", [None, 84, 84, 4])

    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "VALID") + b_conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides = [1, 2, 2, 1], padding = "VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

def trainGraph(inp, out, sess):
    argmax = tf.placeholder("float", [None, ACTIONS]) #to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    gt = tf.placeholder("float", [None]) #ground truth

    action = tf.reduce_sum(tf.mul(out, argmax), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(action - gt))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game = pong.PongGame()
    
    D = deque()

    #intial frame
    frame = game.getPresentFrame()
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    #saver
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON
    
    while(1):
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
        argmax_t = np.zeros([ACTIONS])
        if(random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1
        
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        reward_t, frame = game.getNextFrame(argmax_t)
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
        
        D.append((inp_t, argmax_t, reward_t, inp_t1))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
        
            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]
        
            gt_batch = []
            out_batch = out.eval(feed_dict = {inp : inp_t1_batch})
            
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            train_step.run(feed_dict = {
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch
                           })
        
        inp_t = inp_t1
        t = t+1

        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step = t)

        print "TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t)


def main():
    print "Start"
    sess = tf.InteractiveSession()
    inp, out = createGraph()
    print "Created Graph"
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()
