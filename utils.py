import tensorflow as tf

initializer = tf.contrib.layers.xavier_initializer()
#initializer = tf.contrib.layers.variance_scaling_initializer(factor = 1.0)


def conv(inputs,filters,name):
    net = tf.layers.conv2d(inputs = inputs,
                           filters = filters,
                           kernel_size = [3,3],
                           strides = (1,1),
                           padding ="SAME",
                           kernel_initializer = initializer,
                           name = name,
                           reuse = tf.AUTO_REUSE)
    return net

def maxpool(input,name):
    net = tf.nn.max_pool(value = input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = name)
    return net

def bn(inputs,is_training,name):
    net = tf.contrib.layers.batch_norm(inputs, decay = 0.9, is_training = is_training, reuse = tf.AUTO_REUSE, scope = name)
    return net

def leaky(input):
    return tf.nn.leaky_relu(input)

def relu(input):
    return tf.nn.relu(input)

def drop_out(input, keep_prob):

    return tf.nn.dropout(input, keep_prob)
def dense(inputs, units, name):
    net = tf.layers.dense(inputs = inputs,
                          units = units,
                          reuse = tf.AUTO_REUSE,
                          name = name,
                          kernel_initializer = initializer)
    return net

user_flags = []

def DEFINE_string(name, default_value, doc_string):
    tf.app.flags.DEFINE_string(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)

def DEFINE_integer(name, default_value, doc_string):
    tf.app.flags.DEFINE_integer(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)

def DEFINE_float(name, defualt_value, doc_string):
    tf.app.flags.DEFINE_float(name, defualt_value, doc_string)
    global user_flags
    user_flags.append(name)

def DEFINE_boolean(name, default_value, doc_string):
    tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)

def print_user_flags(line_limit = 100):
    print("-" * 80)

    global user_flags
    FLAGS = tf.app.flags.FLAGS

    for flag_name in sorted(user_flags):
        value = "{}".format(getattr(FLAGS, flag_name))
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        print(log_string)

    return FLAGS