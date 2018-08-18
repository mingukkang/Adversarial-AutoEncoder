## Adversarial AutoEncoder(AAE)- Tensorflow

**I Write the Tensorflow Code for Supervised AAE and SemiSupervised AAE**

## Enviroment
- OS: Ubuntu 16.04

- Graphic Card /RAM : 1080TI /16G

- Python 3.5

- Tensorflow-gpu version:  1.4.0rc2 

- OpenCV 3.4.1

## Schematic of AAE

### Supervised AAE

<img src="Image/Supervised_AAE.png" alt="Drawing" width= "500px"/>

***

### SemiSupervised AAE

<img src="Image/Semisupervised_AAE.png" alt="Drawing" width= "600px"/>

## Code

**Supervised Encoder
```python
def sup_encoder(self, X, keep_prob): # encoder for supervised AAE
    
    with tf.variable_scope("sup_encoder", reuse = tf.AUTO_REUSE):
        net = drop_out(relu(dense(X, self.super_n_hidden, name = "dense_1")), keep_prob)
        net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
        net = dense(net, self.n_z, name ="dense_3")
    
    return net
```
