import tensorflow as tf
from utils import *
from data_utils import *
from prior import *

class AAE:
    def __init__(self, conf, shape, n_labels):
        self.conf = conf
        self.mode = conf.mode
        self.data = conf.data
        self.super_n_hidden = conf.super_n_hidden
        self.semi_n_hidden = conf.semi_n_hidden
        self.n_z = conf.n_z
        self.batch_size = conf.batch_size
        self.prior = conf.prior
        self.w = shape[1]
        self.h = shape[2]
        self.c = shape[3]
        self.length = self.h * self.w * self.c
        self.n_labels = n_labels

    def sup_encoder(self, X, keep_prob): # encoder for supervised AAE

        with tf.variable_scope("sup_encoder", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(X, self.super_n_hidden, name = "dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
            net = dense(net, self.n_z, name ="dense_3")

        return net

    def sup_decoder(self, Z, keep_prob): # decoder for supervised AAE

        with tf.variable_scope("sup_decoder", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(Z, self.super_n_hidden, name = "dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
            net = tf.nn.sigmoid(dense(net, self.length, name = "dense_3"))

        return net

    def discriminator(self,Z, keep_prob): # discriminator for supervised AAE

        with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(Z, self.super_n_hidden, name = "dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.super_n_hidden, name="dense_2")), keep_prob)
            logits = dense(net, 1, name ="dense_3")

        return logits

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

    def semi_encoder(self, X, keep_prob, semi_supervised = False):

        with tf.variable_scope("semi_encoder", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(X, self.semi_n_hidden, name = "dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.semi_n_hidden, name="dense_2")), keep_prob)
            style = dense(net, self.n_z, name ="style")

            if semi_supervised is False:
                labels_generated = tf.nn.softmax(dense(net, self.n_labels, name = "labels"))
            else:
                labels_generated = dense(net, self.n_labels, name = "label_logits")

        return style, labels_generated

    def semi_decoder(self, Z, keep_prob):

        with tf.variable_scope("semi_decoder", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(Z, self.semi_n_hidden, name = "dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.semi_n_hidden, name="dense_2")), keep_prob)
            net = tf.nn.sigmoid(dense(net, self.length, name = "dense_3"))

        return net

    def semi_z_discriminator(self,Z, keep_prob):

        with tf.variable_scope("semi_z_discriminator", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(Z, self.semi_n_hidden, name="dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.semi_n_hidden, name="dense_2")), keep_prob)
            logits = dense(net, 1, name="dense_3")

        return logits

    def semi_y_discriminator(self, Y, keep_prob):

        with tf.variable_scope("semi_y_discriminator", reuse = tf.AUTO_REUSE):
            net = drop_out(relu(dense(Y, self.semi_n_hidden, name = "dense_1")), keep_prob)
            net = drop_out(relu(dense(net, self.semi_n_hidden, name="dense_2")), keep_prob)
            logits = dense(net, 1, name = "dense_3")

        return logits

    def Semi_Adversarial_AutoEncoder(self, X, X_noised, labels, labels_cat, keep_prob):

        X_flatten = tf.reshape(X, [-1 , self.length])
        X_noised_flatten = tf.reshape(X_noised, [-1, self.length])

        style, labels_softmax = self.semi_encoder(X_noised_flatten, keep_prob, semi_supervised = False)
        latent_inputs = tf.concat([labels_softmax, style], axis = 1)
        X_generated = self.semi_decoder(latent_inputs, keep_prob)

        if self.prior is "gaussian":
            z_prior = gaussian(self.batch_size,
                               n_labels = self.n_labels,
                               n_dim = self.n_z,
                               use_label_info = False)

        elif self.prior is "gaussian_mixture":
            z_prior = gaussian_mixture(self.batch_size,
                                       n_labels = self.n_labels,
                                       n_dim = self.n_z)

        elif self.prior is "swiss_roll":
            z_prior = swiss_roll(self.batch_size,
                                 n_labels = self.n_labels,
                                 n_dim = self.n_z)
        else:
            print("FLAGS.prior should be [gaussian, gaussian_mixture, swiss_roll]")

        _, labels_generated = self.semi_encoder(X_noised_flatten, keep_prob, semi_supervised = True)

        D_Y_fake = self.semi_y_discriminator(labels_softmax, keep_prob)
        D_Y_real = self.semi_y_discriminator(labels_cat, keep_prob)

        D_Z_fake = self.semi_z_discriminator(style, keep_prob)
        D_Z_real = self.semi_z_discriminator(tf.convert_to_tensor(z_prior), keep_prob)

        negative_loglikelihood = 0.5* tf.reduce_mean(tf.square(X_generated - X_flatten))

        D_loss_y_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Y_real, labels=tf.ones_like(D_Y_real))
        D_loss_y_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Y_fake, labels=tf.zeros_like(D_Y_fake))
        D_loss_y = tf.reduce_mean(D_loss_y_real + D_loss_y_fake)
        D_loss_z_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = D_Z_real, labels = tf.ones_like(D_Z_real))
        D_loss_z_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = D_Z_fake, labels = tf.zeros_like(D_Z_fake))
        D_loss_z = tf.reduce_mean(D_loss_z_real + D_loss_z_fake)


        G_loss_y = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Y_fake, labels=tf.ones_like(D_Y_fake))
        G_loss_z = tf.nn.sigmoid_cross_entropy_with_logits(logits = D_Z_fake, labels = tf.ones_like(D_Z_fake))
        G_loss = tf.reduce_mean(G_loss_y + G_loss_z)

        CE_labels = tf.nn.softmax_cross_entropy_with_logits(logits = labels_generated, labels = labels)
        CE_labels = tf.reduce_mean(CE_labels)


        return style, X_generated, negative_loglikelihood, D_loss_y, D_loss_z, G_loss, CE_labels

    def optim_op(self, loss ,learning_rate, global_step, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).\
            minimize(loss, global_step = global_step, var_list = var_list)
        return optimizer