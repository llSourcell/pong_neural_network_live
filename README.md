# pong_neural_net_live

##Overview
This is the code for the Live [Youtube](https://www.youtube.com/watch?v=Hqf__FlRlzg) session by @Sirajology. In this live session I build
the game of [Pong](http://www.ponggame.org) from scratch. Then I build a [Deep Q Network](https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work) that gets better and better over time through trial and error. The DQN is a convolutional neural network that reads in pixel data from the game and the game score. Using just those 2 parameters, it learns what moves it needs to make to become better.

##Installation


* tensorflow (https://www.tensorflow.org)
* cv2 (https://pypi.python.org/pypi/opencv-python)
* numpy
* random
* collections
* pygame

use [pip](https://pypi.python.org/pypi/pip) to install the dependencies. Tensorflow and cv2 have pip packages but may need to be more manual. Links provided above ^

##Usage 

Run it like this in terminal. The longer you let it run, the better it will get.

```
python RL.py
```

##Credits

This code was by [malreddysid](https://github.com/malreddysid) i've merely wrapped, updated, and documented it. 

