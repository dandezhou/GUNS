import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import math
import scipy.stats as st
import seaborn as sns
from scipy import io
from scipy.stats import ks_2samp
import matlab.engine
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'    #设置当前使用GPU设备仅为1号设备

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

tf.set_random_seed(100)
np.random.seed(100)
eng = matlab.engine.start_matlab()

train_idx_low = 1
train_idx_high = 13
real_noise = []
for train_idx in range(train_idx_low, train_idx_high):
    # dataFile = 'impulse noise/impulse_noise_' + str(train_idx) + '.mat'
    dataFile = 'impulse noise/real/real_noise_20150814_' + str(train_idx) + '.mat'
    data_matlab = io.loadmat(dataFile)
    noise_imp = data_matlab['real_noise']
    real_noise.append(noise_imp)
    # data = np.asarray(data)
real_noise = np.reshape(np.asarray(real_noise), [-1, ])


def generate_real_samples_with_labels(data):  # 正弦波叠加法
    h_p = data.reshape((-1, 1))
    size = np.shape(h_p)

    return h_p


condition_depth = 2
# condition_dim = 4
# Z_dim = 16
realdata_size = 400
Z_rand = 128
Z_dim = Z_rand
Z_DISCRIMINATOR = realdata_size
batch_size_g = 12000  # 信道长度
batch_size_d = int(batch_size_g / realdata_size)
kernal_size = 2
n_kernal_1 = 14
n_kernal_2 = 14
n_outputs = 1

model = 'ChannelGAN_Rayleigh_'
data = generate_real_samples_with_labels(real_noise)

with tf.Graph().as_default() as graph:
    def generator(z):  #  生成器采用的是深度网络
        # z_combine = tf.concat([z, conditioning], 1)
        G_h1 = tf.nn.leaky_relu(tf.matmul(z, G_W1) + G_b1)    #输入的随机噪声乘以G_W1矩阵加上偏置G_b1
        # G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)
        # G_h3 = tf.nn.leaky_relu(tf.matmul(G_h2, G_W3) + G_b3)
        # G_h3 = tf.layers.batch_normalization(G_h3)
        G_logit = tf.matmul(G_h1, G_W4) + G_b4
        return G_logit


    stats_graph(graph)

    def discriminator(X):  #  判别器采用的是深度网络
        D_h1_real = tf.nn.leaky_relu(tf.nn.conv1d(X, D_W1, 1, padding='SAME')) # 第一层卷积
        D_h1_real = tf.nn.pool(input=D_h1_real, window_shape=[2], strides=[2], pooling_type="MAX", padding="SAME") # 池化\
        # D_h1_real = tf.layers.batch_normalization(D_h1_real)  # 归一化
        D_h2_real = tf.nn.leaky_relu(tf.nn.conv1d(D_h1_real, D_W2, 1, padding='SAME')) #第二层卷积
        D_h2_real = tf.nn.pool(input=D_h2_real, window_shape=[2], strides=[2], pooling_type="MAX", padding="SAME") #池化
        D_h2_real = tf.reshape(D_h2_real, [-1, D_W3.get_shape().as_list()[0]]) #变形
        # D_h2_real = tf.layers.batch_normalization(D_h2_real) #归一化
        D_logit = tf.nn.leaky_relu(tf.matmul(D_h2_real, D_W4) + D_b4) # 全连接层
        D_prob = tf.nn.sigmoid(D_logit)     #用sigmoid函数将变量映射到0~1
        return D_prob, D_logit

    def sample_Z(sample_size):  #高斯噪声
        ''' Sampling the generation noise Z from normal distribution '''

        xavier_stddev = 1.0
        # return np.random.normal(loc=0,scale=xavier_stddev,size=sample_size)   #从指定正态分布中输出随机值

        return np.random.uniform(-0.8, 0.8, size=sample_size)

    def xavier_init(size):  #初始化参数
        in_dim = size[0]
        # xavier_stddev = 1. / tf.sqrt(in_dim / 2.)   #初始化标准差
        xavier_stddev = 0.3
        return tf.random_normal(shape=size, stddev=xavier_stddev)   #从指定正态分布中输出随机值

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.2))


    def combine_complex(x, len_x):  # 将实虚实虚结合成复数
        even_number_x = np.arange(0, len_x, 2)
        odd_number_x = np.arange(1, len_x, 2)
        real_x = tf.transpose(tf.gather(tf.transpose(x), even_number_x))
        imag_x = tf.transpose(tf.gather(tf.transpose(x), odd_number_x))  # 实虚交替信号
        x_complex = tf.complex(real_x, imag_x)
        return x_complex

    sess = tf.Session()     #会话层，用来激活下一行（）中的product
    sess.run(tf.global_variables_initializer())     #初始化

    # data = noise_imp
    # D_W1 = tf.Variable(xavier_init([2 + condition_dim, 48]))

    D_W1 = init_weights([kernal_size, 1, n_kernal_1])
    D_W2 = init_weights([kernal_size, n_kernal_1, n_kernal_2])
    D_W3 = init_weights([n_kernal_2 * 25 * batch_size_d, 1])
    D_W4 = tf.Variable(xavier_init([n_kernal_2 * 25 * batch_size_d, 1]))
    D_b4 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_W3, D_W4, D_b4]

    G_W1 = tf.Variable(xavier_init([Z_dim, 100]))
    G_b1 = tf.Variable(tf.zeros(shape=[100]))
    G_W2 = tf.Variable(xavier_init([100, 100]))
    G_b2 = tf.Variable(tf.zeros(shape=[100]))
    G_W3 = tf.Variable(xavier_init([100, 100]))
    G_b3 = tf.Variable(tf.zeros(shape=[100]))
    G_W4 = tf.Variable(xavier_init([100, 1]))
    G_b4 = tf.Variable(tf.zeros(shape=[1]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]
    R_sample = tf.placeholder(tf.float32, shape=[None, 1])  #真实样本
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])     #生成器输入
    # Condition = tf.placeholder(tf.float32, shape=[None, condition_dim])
    G_sample = generator(Z)      #生成器输出

    # 设置判别器的条件输入
    G_sample_r = tf.reshape(G_sample, [-1, realdata_size,1], name=None)
    R_sample_r = tf.reshape(R_sample, [-1, realdata_size,1], name=None)

    D_prob_real, D_logit_real = discriminator(R_sample_r)      #判别器判别的真实数据的结果
    D_prob_fake, D_logit_fake = discriminator(G_sample_r)      #判别器判别的生成数据的结果

    D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)    #判别器的误差
    G_loss = -1 * tf.reduce_mean(D_logit_fake)      #生成器的误差,将判别器返回的对生成数据的判别结果与1作比较
    lr_d = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])

    D_solver = tf.train.AdamOptimizer(learning_rate=lr_d, beta1=0.5).minimize(D_loss, var_list=theta_D)#D训练器
    G_solver = tf.train.AdamOptimizer(learning_rate=lr_G, beta1=0.5).minimize(G_loss, var_list=theta_G)#G训练器

    save_fig_path = 'imag_2'
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    i = 0

    plot_every = 10000
    xmax = 4
    time = 0

    # 声明要查看的标量
    tf.summary.scalar("loss", D_loss)
    merge_summary = tf.summary.merge_all()

    saver = tf.train.Saver()    #模型保存器
    saving_name = 'model' + '/noise_generator'
    saving_path = 'model/'
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # ckpt = tf.train.get_checkpoint_state(saving_path)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        writer = tf.summary.FileWriter("tf.log", sess.graph)
        lr_dvalue = 1e-5
        lr_gvalue = 1e-5
        flag = 0
        for it in range(30001):
            # if it % 30001 == 0:
            #     lr_value = lr_value/2

            start_idx = it * batch_size_d * realdata_size % len(data)
            # start_idx = it * batch_size
            if start_idx + batch_size_d >= len(data):
                continue

            # nn = np.random.randint(len(data) - 1, size=int(batch_size))
            X_mb = data[start_idx:start_idx + realdata_size * batch_size_d, :]
            # X_mb = data[nn, :]

            samples_component = sess.run(G_sample, feed_dict={lr_d: lr_dvalue,
                                                              lr_G: lr_gvalue,
                                                              Z: sample_Z((batch_size_g, Z_rand))})

            for d_idx in range(100):
                if it > 4000 \
                        and abs(D_loss_curr) < 10:
                    #     and it % 10 == 0:
                    flag += 1
                    # lr_gvalue = lr_gvalue / 10
                    # lr_dvalue = lr_dvalue / 10
                    # if flag == 2:
                    break
                    # lr_gvalue = lr_gvalue / 20
                    # lr_dvalue = lr_dvalue / 1000

                _, D_loss_curr = sess.run([D_solver, D_loss],
                                          feed_dict={lr_d: lr_dvalue,
                                                     lr_G: lr_gvalue,
                                                     R_sample: X_mb,
                                                     Z: sample_Z((batch_size_g, Z_rand))
                                                     })
            if flag == 1:
                print(it)
                print('D_loss: ', D_loss_curr)

                saver.save(sess, saving_name)

                env = np.reshape(samples_component, [-1, ])

                fig_2 = plt.figure()
                sns.kdeplot(env)
                plt.savefig(save_fig_path + '/' + str(it) + '_fuzhi_pdf', )
                break

            merge_result = sess.run(merge_summary,
                                    feed_dict={lr_d: lr_dvalue,
                                               lr_G: lr_gvalue,
                                               R_sample: X_mb,
                                               Z: sample_Z((batch_size_g, Z_rand))
                                               })
            writer.add_summary(merge_result, it)

            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={lr_d: lr_dvalue,
                                                 lr_G: lr_gvalue,
                                                 R_sample: X_mb,
                                                 Z: sample_Z((batch_size_g, Z_rand))
                                                 })

            if it % 500 == 0:
                print(it)
                # time = time+1
                print('D_loss: ', D_loss_curr)

                saver.save(sess, saving_name)
            # if it % 1000 == 0:

                env = np.reshape(samples_component,[-1,])

                # fig_1 = plt.figure()
                # plt.stem(env, ls="-", lw=2, label="plot figure")
                # plt.savefig(save_fig_path + '/' + str(it) + '_xiangying', )
                fig_2 = plt.figure()
                sns.kdeplot(env)
                plt.savefig(save_fig_path + '/' + str(it) + '_fuzhi_pdf',)


