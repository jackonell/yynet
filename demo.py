import tensorflow as tf

class Vgg16:

    def build(self,rgb,train_mode = False):
        '''
            定义vgg16
        :param rgb 输入为224X224RGB图像
        :param train_mode 标识符，如果处于训练阶段，则dropout会打开
        '''
        self.conv1_1 = self.conv_layer(rgb,"conv1_1",3,64)
        self.conv1_2 = self.conv_layer(self.conv1_1,"conv1_2",64,64)
        self.pool1 = self.max_pool(self.conv1_2,"pool1")

        self.conv2_1 = self.conv_layer(self.pool1,"conv2_1",64,128)
        self.conv2_2 = self.conv_layer(self.conv2_1,"conv2_2",128,128)
        self.pool2 = self.max_pool(self.conv2_2,"pool2")

        self.conv3_1 = self.conv_layer(self.pool2,"conv3_1",128,256)
        self.conv3_2 = self.conv_layer(self.conv3_1,"conv3_2",256,256)
        self.conv3_3 = self.conv_layer(self.conv3_2,"conv3_3",256,256)
        self.pool3 = self.max_pool(self.conv3_3,"pool3")

        self.conv4_1 = self.conv_layer(self.pool3,"conv4_1",256,512)
        self.conv4_2 = self.conv_layer(self.conv4_1,"conv4_2",512,512)
        self.conv4_3 = self.conv_layer(self.conv4_2,"conv4_3",512,512)
        self.pool4 = self.conv_layer(self.conv4_3,"pool4")

        self.conv5_1 = self.conv_layer(self.pool4,"conv5_1",512,512)
        self.conv5_2 = self.conv_layer(self.conv5_1,"conv5_2",512,512)
        self.conv5_3 = self.conv_layer(self.conv5_2,"conv5_3",512,512)
        self.pool5 = self.max_pool(self.conv5_3,"pool5")

        self.fc6 = self.fc_layer(self.pool5,"fc6",25088,4096)    #25088 = ((224//(2**5))**2)*512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode:
            self.relu6 = tf.nn.dropout(self.relu6,self.dropout)

        self.fc7 = self.fc_layer(self.relu6,"fc7",4096,4096)
        self.relu7 = tf.relu(self.fc7)
        if train_mode:
            self.relu7 = tf.nn.dropout(self.relu7,self.dropout)

        self.fc8 = self.fc_layer(self.relu7,"fc8",4096,1000);

        self.prob = tf.nn.softmax(self.fc8,name="prob")

    '''
      卷积例子如下：
      with tf.variable_scope('layer1-conv1'):
            w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]),name='weight')
            b_c1 = tf.Variable(b_alpha*tf.random_normal([32]),name='bias')
            relu1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    '''
    def conv_layer(self,pre_layer, name , in_channels, out_channels,filter_size=3):
        with tf.variable_scope(name):
            filter = tf.truncated_normal([filter_size,filter_size,in_channels,out_channels],0.0,0.001)
            w_c = tf.Variable(filter,name='weight')

            bs = tf.truncated_normal([out_channels],0.0,0.001)
            b_c = tf.Variable(bs,name='bias');

            conv = tf.nn.conv2d(pre_layer,w_c,[1,1,1,1],padding='SAME')
            bias = tf.nn.bias_add(conv,b_c)
            relu = tf.nn.relu(bias)

            return relu

    def max_pool(self,pre_layer,name):
        with tf.variable_scope(name):
            tf.nn.max_pool(pre_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    def fc_layer(self,pre_layer,name,in_size,out_size):
        with tf.variable_scope(name):
            inc = tf.truncated_normal([in_size,out_size],0.0,0.001)
            w_f = tf.Variable(inc,name='weight')

            outc = tf.truncated_normal([out_size],0.0,0.001)
            b_f = tf.Variable(outc,name='bias')

            x = tf.reshape(pre_layer,[-1,in_size])
            mul = tf.matmul(x,w_f)
            fc = tf.nn.bias_add(mul,b_f)

            return fc
