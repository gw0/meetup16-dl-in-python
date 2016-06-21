meetup16-dl-in-python
=====================

Examples and code used in the presentation *"Deep learning in Python?"* at *Meetup Deep Learning Ljubljana* on 2016-06-20.

> We will dive into popular Python deep learning frameworks such as Theano, TensorFlow, and Keras and demonstrate a few hands-on examples from classic machine learning, computer vision, and natural language processing. After comparing the resulting code of dense, convolutional (CNN), and recurrent neural networks (RNN), we will take a glimpse at how Keras works under the hood. Additionally we will see how to kick-start your deep learning research and move to production using Docker containers.


Usage
-----

Requirements:

- *Docker* to use already prepared container images ([`gw000/keras-full`](http://gw.tnode.com/docker/keras-full/) or [`gw000/keras`](http://gw.tnode.com/docker/keras/))
- (or alternatively an environment with *Keras*, *Theano*, and *TensorFlow*)

Using *Docker* container [`gw000/keras-full`](http://gw.tnode.com/docker/keras-full/) with *Keras*, *Theano*, *TensorFlow*, *Python 2 and 3*, and the *Jupyter Notebook* web interface (<http://localhost:8888/>) on CPU:

```bash
$ docker run -it -p=8888:8888 gw000/keras-full
```

Using *Docker* container [`gw000/keras-full`](http://gw.tnode.com/docker/keras-full/) on GPU with notebooks in `/srv/notebooks/`:

```bash
$ docker run -it $(ls /dev/nvidia* | xargs -I{} echo '--device={}') -p=8888:8888 -v=/srv/notebooks:/srv gw000/keras-full
```


License
=======

Copyright &copy; 2016 *gw0* [<http://gw.tnode.com/>] &lt;<gw.2016@tnode.com>&gt;

Copyright &copy; 2015 *François Chollet*, *Google, Inc.* and respective contributors to some examples in [Keras](http://github.com/fchollet/keras)

This code is licensed under the [MIT license](LICENSE_MIT.txt) (*MIT*).
