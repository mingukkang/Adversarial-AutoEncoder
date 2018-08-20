import tensorflow as tf
import numpy as np
import time
from utils import *
from plot import *
from AAE import *

flags = tf.app.flags

flags.DEFINE_string("mode", "semi_supervised", "[supervised | semi_supervised]")
flags.DEFINE_string("data", "MNIST", "[MNIST | CIFAR_10]")
flags.DEFINE_string("prior", "gaussian", "[gaussain | gaussain_mixture | swiss_roll]")


flags.DEFINE_integer("super_n_hidden", 3000, "the number of elements for hidden layers")
flags.DEFINE_integer("semi_n_hidden", 3000, "teh number of elements for hidden layers")
flags.DEFINE_integer("n_epoch", 150, "number of Epoch for training")
flags.DEFINE_integer("n_z", 2, "Dimension of Latent variables")
flags.DEFINE_integer("num_samples",5000, "number of samples for semi supervised learning")
flags.DEFINE_integer("batch_size", 128, "Batch Size for training")

flags.DEFINE_float("keep_prob", 0.9, "dropout rate")
flags.DEFINE_float("lr_start", 0.001, "initial learning rate")
flags.DEFINE_float("lr_mid", 0.0005, "mid learning rate")
flags.DEFINE_float("lr_end", 0.0001, "final learning rate")

flags.DEFINE_bool("noised", True, "")
flags.DEFINE_bool("PMLR", True, "Boolean for plot manifold learning result")
flags.DEFINE_bool("PARR", True, "Boolean for plot analogical reasoning result")

conf = flags.FLAGS

if conf.mode is "supervised":

    data_pipeline = data_pipeline(conf.data)

    train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = data_pipeline.load_preprocess_data()

    _, height, width, channel = np.shape(train_xs)
    n_cls = np.shape(train_ys)[1]

    X = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name="Inputs")
    X_noised = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name="Inputs_noised")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name="Input_labels")
    z_prior = tf.placeholder(tf.float32, shape=[None, conf.n_z], name="z_prior")
    z_id = tf.placeholder(tf.float32, shape = [None, n_cls], name = "prior_labels")
    latent = tf.placeholder(tf.float32, shape = [None, conf.n_z], name = "latent_for_generation")
    keep_prob = tf.placeholder(dtype = tf.float32, name = "dropout_rate")
    lr_ = tf.placeholder(dtype = tf.float32, name = "learning_rate")
    global_step = tf.Variable(0, trainable=False)

    AAE = AAE(conf, [_, height, width, channel], n_cls)
    z_generated, X_generated, negative_log_likelihood, D_loss, G_loss = AAE.Sup_Adversarial_AutoEncoder(X,
                                                                                                        X_noised,
                                                                                                        Y,
                                                                                                        z_prior,
                                                                                                        z_id,
                                                                                                        keep_prob)
    images_PMLR = AAE.sup_decoder(latent, keep_prob)
    total_batch = data_pipeline.get_total_batch(train_xs, conf.batch_size)

    total_vars = tf.trainable_variables()
    var_AE = [var for var in total_vars if "encoder" or "decoder" in var.name]
    var_generator = [var for var in total_vars if "encoder" in var.name]
    var_discriminator = [var for var in total_vars if "discriminator" in var.name]

    op_AE = tf.train.AdamOptimizer(learning_rate = lr_).minimize(negative_log_likelihood,
                                                                global_step = global_step,
                                                                var_list = var_AE)

    op_D = tf.train.AdamOptimizer(learning_rate = lr_/5). minimize(D_loss,
                                                                global_step = global_step,
                                                                var_list = var_discriminator)
    op_G = tf.train.AdamOptimizer(learning_rate = lr_).minimize(G_loss,
                                                               global_step = global_step,
                                                               var_list = var_generator)

    batch_t_xs, batch_tn_xs, batch_t_ys = data_pipeline.next_batch(valid_xs, valid_ys, 100, make_noise= False)
    data_pipeline.initialize_batch()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    start_time  = time.time()
    for i in range(conf.n_epoch):
        likelihood = 0
        D_value = 0
        G_value = 0
        for j in range(total_batch):
            batch_xs, batch_noised_xs, batch_ys = data_pipeline.next_batch(train_xs,
                                                                           train_ys,
                                                                           conf.batch_size,
                                                                           make_noise=conf.noised)
            if conf.prior is "gaussian":
                z_prior_, z_id_ = gaussian(conf.batch_size,
                                         n_labels = n_cls,
                                         n_dim = conf.n_z,
                                         use_label_info = True)
                z_id_onehot = np.eye(n_cls)[z_id_].astype(np.float32)

            elif conf.prior is "gaussian_mixture":
                z_id_ = np.random.randint(0, n_cls, size=[conf.batch_size])
                z_id_onehot = np.eye(n_cls)[z_id_].astype(np.float32)
                z_prior_ = gaussian_mixture(conf.batch_size,
                                           n_labels = n_cls,
                                           n_dim = conf.n_z,
                                           label_indices = z_id_)

            elif conf.prior is "swiss_roll":
                z_id_ = np.random.randint(0, n_cls, size=[conf.batch_size])
                z_id_onehot = np.eye(n_cls)[z_id_].astype(np.float32)
                z_prior_ = swiss_roll(conf.batch_size,
                                     n_labels = n_cls,
                                     n_dim = conf.n_z,
                                     label_indices = z_id_)
            else:
                print("FLAGS.prior should be [gaussian, gaussian_mixture, swiss_roll]")

            if i <= 50:
                lr_value = conf.lr_start
            elif i <=100:
                lr_value = conf.lr_mid
            else:
                lr_value = conf.lr_end

            feed_dict = {X: batch_xs,
                         X_noised: batch_noised_xs,
                         Y: batch_ys,
                         z_prior: z_prior_,
                         z_id: z_id_onehot,
                         lr_: lr_value,
                         keep_prob: conf.keep_prob}

            # AutoEncoder phase
            l, _, g = sess.run([negative_log_likelihood, op_AE, global_step], feed_dict=feed_dict)

            # Discriminator phase
            l_D, _ = sess.run([D_loss, op_D], feed_dict = feed_dict)

            l_G, _ = sess.run([G_loss, op_G], feed_dict = feed_dict)

            likelihood += l/total_batch
            D_value += l_D/total_batch
            G_value += l_G/total_batch

        if i % 5 == 0 or i == (conf.n_epoch -1):
            images = sess.run(X_generated, feed_dict = {X:batch_t_xs,
                                                        X_noised: batch_tn_xs,
                                                        keep_prob: 1.0})
            images = np.reshape(images, [-1, height, width, channel])
            name = "Manifold_canvas_" + str(i)
            plot_manifold_canvas(images, 10, type = "MNIST", name = name)


        hour = int((time.time() - start_time) / 3600)
        min = int(((time.time() - start_time) - 3600 * hour) / 60)
        sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
        print("Epoch: %3d   lr_AE: %.5f   loss_AE: %.4f   Time: %d hour %d min %d sec" % (i, lr_value, likelihood, hour, min, sec))
        print("             lr_D: %.5f   loss_D: %.4f"  % (lr_value/5, D_value))
        print("             lr_G: %.5f   loss_G: %.4f\n" % (lr_value, G_value))

    ## code for 2D scatter plot
    if conf.n_z == 2:
        print("-" * 80)
        print("plot 2D Scatter Result")
        test_total_batch = data_pipeline.get_total_batch(test_xs, 128)
        data_pipeline.initialize_batch()
        latent_holder = []
        for i in range(test_total_batch):
            batch_test_xs, batch_test_noised_xs, batch_test_ys = data_pipeline.next_batch(test_xs,
                                                                                          test_ys,
                                                                                          conf.batch_size,
                                                                                          make_noise=False)
            feed_dict = {X: batch_test_xs,
                         X_noised: batch_test_noised_xs,
                         keep_prob: 1.0}

            latent_vars = sess.run(z_generated, feed_dict=feed_dict)
            latent_holder.append(latent_vars)
        latent_holder = np.concatenate(latent_holder, axis=0)
        plot_2d_scatter(latent_holder[:, 0], latent_holder[:, 1], test_ys[:len(latent_holder)])

    if conf.PMLR is True:
        print("-" * 80)
        assert conf.n_z == 2, "Error: n_z should be 2"
        print("plot Manifold Learning Result")
        x_axis = np.linspace(-0.5, 0.5, 10)
        y_axis = np.linspace(0.5, -0.5, 10)
        z_holder = []
        for i, yi in enumerate(y_axis):
            for j, xi in enumerate(x_axis):
                z_holder.append([xi, yi])
        length = len(z_holder)
        MLR = sess.run(images_PMLR, feed_dict={latent: z_holder, keep_prob: 1.0})
        MLR = np.reshape(MLR, [-1, height, width, channel])
        p_name = "PMLR/PMLR"
        plot_manifold_canvas(MLR, 10, "MNIST", p_name)

elif conf.mode is "semi_supervised":

    Data = data_pipeline(conf.data)
    Data_semi = data_pipeline(conf.data)
    train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = Data.load_preprocess_data()
    valid_xs, valid_ys = valid_xs[:conf.num_samples], valid_ys[:conf.num_samples]

    _, height, width, channel = np.shape(train_xs)
    n_cls = np.shape(train_ys)[1]

    X = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name="Input")
    X_noised = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name="Input_noised")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name="Input_labels")
    Y_cat = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name="labels_cat")
    z_prior_ = tf.placeholder(dtype = tf.float32, shape = [None,conf.n_z], name = "z_prior" )
    latent = tf.placeholder(dtype = tf.float32, shape = [None, conf.n_z + n_cls], name = "latent_for_generation")
    keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_rate")
    lr_ = tf.placeholder(dtype=tf.float32, name="learning_rate")
    global_step = tf.Variable(0, trainable=False)

    AAE = AAE(conf, [_, height, width, channel], n_cls)

    style, X_generated, negative_log_likelihood, D_loss_y, D_loss_z, G_loss, CE_labels =AAE.Semi_Adversarial_AutoEncoder(X,
                                                                                                                         X_noised,
                                                                                                                         Y,
                                                                                                                         Y_cat,
                                                                                                                         z_prior_,
                                                                                                                         keep_prob)
    images_PARR = AAE.semi_decoder(latent, keep_prob)
    images_manifold = AAE.semi_decoder(latent, keep_prob)

    total_batch = Data.get_total_batch(train_xs, conf.batch_size)

    total_vars = tf.trainable_variables()
    var_AE = [var for var in total_vars if "encoder" or "decoder" in var.name]
    var_z_discriminator = [var for var in total_vars if "z_discriminator" in var.name]
    var_y_discriminator = [var for var in total_vars if "y_discriminator" in var.name]
    var_generator = [var for var in total_vars if "encoder" in var.name]

    op_AE = tf.train.AdamOptimizer(learning_rate = lr_).minimize(negative_log_likelihood, global_step = global_step, var_list = var_AE)
    op_y_D = tf.train.AdamOptimizer(learning_rate = lr_/5).minimize(D_loss_y, global_step = global_step, var_list = var_y_discriminator)
    op_z_D = tf.train.AdamOptimizer(learning_rate = lr_/5).minimize(D_loss_z, global_step = global_step, var_list = var_z_discriminator)
    op_G = tf.train.AdamOptimizer(learning_rate = lr_).minimize(G_loss, global_step = global_step, var_list = var_generator)
    op_CE_labels = tf.train.AdamOptimizer(learning_rate = lr_).minimize(CE_labels, global_step = global_step, var_list = var_generator)

    batch_t_xs, batch_tn_xs, batch_t_ys = Data.next_batch(valid_xs, valid_ys, 100, make_noise = False)
    Data.initialize_batch()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    start_time = time.time()
    for i in range(conf.n_epoch):
        likelihood = 0
        D_z_value = 0
        D_y_value = 0
        G_value = 0
        CE_value = 0

        if i <= 50:
            lr_value = conf.lr_start
        elif i <= 100:
            lr_value = conf.lr_mid
        else:
            lr_value = conf.lr_end

        for j in range(total_batch):
            batch_xs, batch_noised_xs, batch_ys = Data.next_batch(train_xs,
                                                                  train_ys,
                                                                  conf.batch_size,
                                                                  make_noise = conf.noised)

            real_cat_labels = np.random.randint(low = 0, high = n_cls, size = conf.batch_size)
            real_cat_labels = np.eye(n_cls)[real_cat_labels]

            if conf.prior is "gaussian":
                z_prior = gaussian(conf.batch_size,
                                   n_labels=n_cls,
                                   n_dim=conf.n_z,
                                   use_label_info=False)

            elif conf.prior is "gaussian_mixture":
                z_prior = gaussian_mixture(conf.batch_size,
                                           n_labels=n_cls,
                                           n_dim=conf.n_z)

            elif conf.prior is "swiss_roll":
                z_prior = swiss_roll(conf.batch_size,
                                     n_labels=n_cls,
                                     n_dim=conf.n_z)
            else:
                print("FLAGS.prior should be [gaussian, gaussian_mixture, swiss_roll]")

            feed_dict = {X: batch_xs,
                         X_noised: batch_noised_xs,
                         Y: batch_ys,
                         Y_cat: real_cat_labels,
                         z_prior_: z_prior,
                         lr_: lr_value,
                         keep_prob: conf.keep_prob}

            # AutoEncoder phase
            l, _, g = sess.run([negative_log_likelihood, op_AE, global_step], feed_dict = feed_dict)

            # z_Discriminator phase
            l_z_D,_ = sess.run([D_loss_z, op_z_D], feed_dict = feed_dict)

            # y_Discriminator phase
            l_y_D, _ = sess.run([D_loss_y, op_y_D], feed_dict=feed_dict)

            # Generator phase
            l_G, _ = sess.run([G_loss, op_G], feed_dict = feed_dict)

            batch_semi_xs, batch_noised_semi_xs,batch_semi_ys = Data_semi.next_batch(valid_xs,
                                                                                     valid_ys,
                                                                                     conf.batch_size,
                                                                                     make_noise = False)

            feed_dict = {X: batch_semi_xs,
                         X_noised: batch_noised_semi_xs,
                         Y: batch_semi_ys,
                         Y_cat: real_cat_labels,
                         lr_:lr_value,
                         keep_prob: conf.keep_prob}

            # Cross_Entropy phase
            CE, _ = sess.run([CE_labels, op_CE_labels], feed_dict = feed_dict)

            likelihood += l/total_batch
            D_z_value += l_z_D/total_batch
            D_y_value += l_y_D/total_batch
            G_value += l_G/total_batch
            CE_value += CE/total_batch

        if i % 5 == 0 or i == (conf.n_epoch -1):
            images = sess.run(X_generated, feed_dict = {X:batch_t_xs,
                                                        X_noised: batch_tn_xs,
                                                        keep_prob: 1.0})
            images = np.reshape(images, [-1, height, width, channel])
            name = "Manifold_semi_canvas_" + str(i)
            plot_manifold_canvas(images, 10, type = "MNIST", name = name)


        hour = int((time.time() - start_time) / 3600)
        min = int(((time.time() - start_time) - 3600 * hour) / 60)
        sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
        print("Epoch: %3d   lr_AE_G_CE: %.5f   lr_D: %.5f   Time: %d hour %d min %d sec" % (i, lr_value,lr_value/5, hour, min, sec))
        print("loss_AE: %.5f" % (likelihood))
        print("loss_z_D: %.4f   loss_y_D: %f"  % (D_z_value, D_y_value))
        print("loss_G: %.4f   CE_semi: %.4f\n" % (G_value, CE_value))

    if conf.PARR is True:
        print("-"*80)
        print("plot analogical reasoning result")
        z_holder = []
        for i in range(n_cls):
            z_ = np.random.rand(10, conf.n_z)
            z_holder.append(z_)
        z_holder = np.concatenate(z_holder, axis = 0)
        y = [j for j in range(n_cls)]
        y = y*10
        length = len(z_holder)
        y_one_hot = np.zeros((length, n_cls))
        y_one_hot[np.arange(length), y] = 1
        y_one_hot = np.reshape(y_one_hot, [-1, n_cls])
        z_concated = np.concatenate([z_holder, y_one_hot], axis=1)
        PARR = sess.run(images_PARR, feed_dict = {latent: z_concated, keep_prob: 1.0})
        PARR = np.reshape(PARR, [-1, height, width, channel])
        p_name = "PARR/Cond_generation"
        plot_manifold_canvas(PARR, 10, "MNIST", p_name)

    ## code for 2D scatter plot
    if conf.n_z == 2:
        print("-" * 80)
        print("plot 2D Scatter Result")
        test_total_batch = Data.get_total_batch(test_xs, 128)
        Data.initialize_batch()
        latent_holder = []
        for i in range(test_total_batch):
            batch_test_xs, batch_test_noised_xs, batch_test_ys = Data.next_batch(test_xs,
                                                                                 test_ys,
                                                                                 conf.batch_size,
                                                                                 make_noise=False)
            feed_dict = {X: batch_test_xs,
                         X_noised: batch_test_noised_xs,
                         Y: batch_test_ys,
                         keep_prob: 1.0}

            latent_vars = sess.run(style, feed_dict=feed_dict)
            latent_holder.append(latent_vars)
        latent_holder = np.concatenate(latent_holder, axis=0)
        plot_2d_scatter(latent_holder[:, 0], latent_holder[:, 1], test_ys[:len(latent_holder)])

    if conf.PMLR is True:
        print("-"*80)
        assert conf.n_z == 2, "Error: n_z should be 2"
        print("plot Manifold Learning Results")
        x_axis = np.linspace(-0.5,0.5,10)
        y_axis = np.linspace(-0.5,0.5,10)
        z_holder = []
        for i,xi in enumerate(x_axis):
            for j, yi in enumerate(y_axis):
                z_holder.append([xi,yi])
        length = len(z_holder)
        for k in range(n_cls):
            y = [k]*length
            y_one_hot = np.zeros((length, n_cls))
            y_one_hot[np.arange(length), y] = 1
            y_one_hot = np.reshape(y_one_hot, [-1,n_cls])
            z_concated = np.concatenate([z_holder, y_one_hot], axis=1)
            MLR = sess.run(images_manifold, feed_dict = {latent: z_concated, keep_prob: 1.0})
            MLR = np.reshape(MLR, [-1, height, width, channel])
            p_name = "PMLR/labels" +str(k)
            plot_manifold_canvas(MLR, 10, "MNIST", p_name)

if __name__ =='__main__':
    tf.app.run()
