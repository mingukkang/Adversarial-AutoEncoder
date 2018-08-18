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

**Supervised Encoder**
```python
def sup_encoder(self, X, keep_prob): # encoder for supervised AAE
    
    with tf.variable_scope("sup_encoder", reuse = tf.AUTO_REUSE):
        net = drop_out(relu(dense(X, self.super_n_hidden, name = "dense_1")), keep_prob)
        net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
        net = dense(net, self.n_z, name ="dense_3")
    
    return net
```

**Supervised Decoder**
```python
def sup_decoder(self, Z, keep_prob): # decoder for supervised AAE

    with tf.variable_scope("sup_decoder", reuse = tf.AUTO_REUSE):
        net = drop_out(relu(dense(Z, self.super_n_hidden, name = "dense_1")), keep_prob)
        net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
        net = tf.nn.sigmoid(dense(net, self.length, name = "dense_3"))

    return net
```

**Supervised Discriminator**
```python
def discriminator(self,Z, keep_prob): # discriminator for supervised AAE

    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
        net = drop_out(relu(dense(Z, self.super_n_hidden, name = "dense_1")), keep_prob)
        net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
        logits = dense(net, 1, name ="dense_3")

        return logits
```

**Supervised Adversarial AutoEncoder**
```python
def Sup_Adversarial_AutoEncoder(self, X, X_noised, Y, keep_prob):

    X_flatten = tf.reshape(X, [-1, self.length])
    X_flatten_noised = tf.reshape(X_noised, [-1, self.length])

    z_generated = self.sup_encoder(X_flatten_noised, keep_prob)
    X_generated = self.sup_decoder(z_generated, keep_prob)

    negative_log_likelihood = tf.reduce_mean(tf.square(X_generated - X_flatten))*0.5

    if self.prior is "gaussian":
        z_prior, z_id = gaussian(self.batch_size,
                                 n_labels = self.n_labels,
                                 n_dim = self.n_z,
                                 use_label_info = True)
        z_id_onehot = np.eye(self.n_labels)[z_id].astype(np.float32)

    elif self.prior is "gaussian_mixture":
        z_id = np.random.randint(0, self.n_labels, size=[self.batch_size])
        z_id_onehot = np.eye(self.n_labels)[z_id].astype(np.float32)
        z_prior = gaussian_mixture(self.batch_size,
                                   n_labels = self.n_labels,
                                   n_dim = self.n_z,
                                   label_indices = z_id)

    elif self.prior is "swiss_roll":
        z_id = np.random.randint(0, self.n_labels, size=[self.batch_size])
        z_id_onehot = np.eye(self.n_labels)[z_id].astype(np.float32)
        z_prior = swiss_roll(self.batch_size,
                             n_labels = self.n_labels,
                             n_dim = self.n_z,
                             label_indices = z_id)
    else:
        print("FLAGS.prior should be [gaussian, gaussian_mixture, swiss_roll]")

    z_prior = tf.concat([z_prior, z_id_onehot], axis = 1)
    z_fake = tf.concat([z_generated, Y], axis = 1)
    D_real_logits = self.discriminator(z_prior, keep_prob)
    D_fake_logits = self.discriminator(z_fake, keep_prob)

    D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.zeros_like(D_fake_logits))
    D_loss_true = tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logits, labels = tf.ones_like(D_real_logits))

    G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.ones_like(D_fake_logits))

    D_loss = tf.reduce_mean(D_loss_fake + D_loss_true)
    G_loss = tf.reduce_mean(G_loss)

    return z_generated, X_generated, negative_log_likelihood, D_loss, G_loss
```
