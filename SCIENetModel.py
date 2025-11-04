from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization,\
    Activation, Dropout, Softmax, Permute, Lambda,DepthwiseConv2D,LayerNormalization,Dense,GlobalAveragePooling2D
import keras
import keras.backend as K
from keras.layers import Reshape
from sklearn import preprocessing
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()
from keras.layers import Layer
from tensorflow_probability import layers # 导入TensorFlow Probability的layers模块


def _l2norm(inp, dim):
    x = inp
    lambda_suqare = Lambda(lambda x: K.square(x))
    x = lambda_suqare(x)
    # x = x ** 2
    lambda_sum = Lambda(lambda x: K.sum(x, axis=dim))
    x = lambda_sum(x)
    lambda_sqrt = Lambda(lambda x: K.sqrt(x))
    x = lambda_sqrt(x) + 1e-6
    return inp / x


def Residual(input_tensor, filters, kernel_size):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = Activation("relu")(x1)

    x2 = Conv2D(filters, kernel_size, padding='same')(x1)
    x2 = Activation("relu")(x2)

    weight_1 = Lambda(lambda x:x*0.1)
    weight_x2 = weight_1(x2)
    output = keras.layers.add([input_tensor, weight_x2])
    output = Activation("relu")(output)
    return output
    
    
def Ups(input_tensor):
    lambda_D2S = Lambda(lambda x: tf.depth_to_space(x, 2))
    [b, h, w, c] = input_tensor.get_shape().as_list()
    Conv_F = Conv2D(c*2, kernel_size=(3, 3), padding='same',use_bias=False)(input_tensor)
    output = lambda_D2S(Conv_F)
    output = Activation("relu")(output)
    return output
 

    
def Downs(input_tensor):
    lambda_S2D = Lambda(lambda x: tf.space_to_depth(x, 2))
    [b, h, w, c] = input_tensor.get_shape().as_list()
    Conv_F = Conv2D(c//2, kernel_size=(3, 3), padding='same',use_bias=False)(input_tensor)
    output = lambda_S2D(Conv_F)
    output = Activation("relu")(output)
    return output    


def get_config(self):
  config = super(MyLayer, self).get_config()
  config.update({"head": self.head})
  return config
  


class MyLayer(Layer):
  def __init__(self, head, **kwargs):
    super(MyLayer, self).__init__(**kwargs)
    self.head = head
  def get_config(self):
    config = super(MyLayer, self).get_config()
    config.update({"head": self.head})
    return config
  @classmethod
  def from_config(cls, config):
    head = config.pop("head")
    return cls(head=head, **config)
  def build(self, input_shape):
    # 在build方法中使用add_weight()方法，创建一个可训练的浮点型变量x，形状为(head,)
    self.x = self.add_weight(shape=(self.head,1,1,1), initializer="ones", trainable=True)
    super(MyLayer, self).build(input_shape)
  def call(self, inputs):
    # 在call方法中使用x变量，对输入进行一些计算
    return inputs * self.x
     
def MDTA(input_tensor, filters, nhead):
    [b, h, w, c] = input_tensor.get_shape().as_list()
    sparse_para=0.5
    EPS=1e-10
    lambda_batchdot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
        # l2 normalize
    lambda_l2nor = Lambda(lambda x: x / (tf.norm(x, ord=2, axis=1, keepdims=True)+EPS))
    lambda_mean = Lambda(lambda x: K.mean(x, axis=-2,keepdims=True))
    lambda_l1nor = Lambda(lambda x: x / (tf.norm(x, ord=1, axis=-1, keepdims=True)+EPS))
    lambda_min = Lambda(lambda x: K.min(x, axis=-2,keepdims=True))
    
    layernorm_1= LayerNormalization()(input_tensor)
    QKV_Pre=Conv2D(filters*3, 1, padding='same',use_bias=False)(layernorm_1)
    QKV_F=DepthwiseConv2D(kernel_size=(3, 3), padding='same',use_bias=False)(QKV_Pre)  #b,w,h,filters*3
    
    Q_F,K_F,V_F=Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(QKV_F)
    
    Q_F_head=K.concatenate(Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': nhead})(Q_F), axis=0)
    K_F_head=K.concatenate(Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': nhead})(K_F), axis=0)
    V_F_head=K.concatenate(Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': nhead})(V_F), axis=0)
       
    Q_matrix=lambda_l2nor(Reshape((h * w, c//nhead))(Q_F_head))
    K_matrix_pemu=Permute((2, 1))(lambda_l2nor(Reshape((h * w, c//nhead))(K_F_head)))
    
    QK=lambda_batchdot([K_matrix_pemu, Q_matrix]) # (b*nhead)*filters//nhead * filters//nhead
    temperature =MyLayer(head=nhead)
    QK = K.concatenate(Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': nhead})(K.expand_dims(QK, axis=0)), axis=0)
    QK= K.squeeze(K.concatenate(Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': nhead})(temperature(QK)), axis=1), axis=0)
    
    QK_SP_softmax2=Softmax(axis=-2)(QK)

    V_matrix=Reshape((h * w, c//nhead))(V_F_head)
    add_matrix=lambda_batchdot([V_matrix, QK_SP_softmax2])
    add_F=Reshape((h , w, c//nhead))(add_matrix)
    add_F=K.concatenate(Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': nhead})(add_F), axis=3)
    
    add_linear=Conv2D(filters, 1, padding='same',use_bias=False)(add_F)    
    output=input_tensor+add_linear
    return output  
    
def GDFN(input_tensor, filters):
    [b, h, w, c] = input_tensor.get_shape().as_list()
    lambda_multiply = Lambda(lambda x: tf.multiply(x[0], x[1]))
    layernorm_1= LayerNormalization()(input_tensor)
    gamma=2.66
    F_Pre_3 = Conv2D(int(filters*gamma)*2, 1, padding='same',use_bias=False)(layernorm_1)
    F_D_3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same',use_bias=False)(F_Pre_3)
    
    UP_F_3,Dwn_F_3=Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 2})(F_D_3)
    
    UP_3_5 = UP_F_3
    Dwn_3_5 = Dwn_F_3
    
    Dwn_Ge_3_5 = Activation('relu')(Dwn_3_5)    
    add_F=lambda_multiply([UP_3_5, Dwn_Ge_3_5])
    add_linear=Conv2D(filters, 1, padding='same',use_bias=False)(add_F)  
    output=input_tensor+add_linear
    return output  

def l1norm_at(inp, dim):
    x=inp
    lambda_abs = Lambda(lambda x: K.abs(x))
    lambda_sum = Lambda(lambda x:K.sum(x,axis=dim))
    x = lambda_abs(x)
    x = lambda_sum(x)+1e-6
    return inp/x


def l2norm_at(inp, dim):
    x=inp
    lambda_square = Lambda(lambda x: K.square(x))
    lambda_sqrt = Lambda(lambda x: K.sqrt(x))
    lambda_sum = Lambda(lambda x:K.sum(x,axis=dim))
    x = lambda_square(x)
    x = lambda_sum(x)
    x = lambda_sqrt(x)+1e-6
    return inp/x

def SDTA(input_tensor, filters, nhead):
    [b, h, w, c] = input_tensor.get_shape().as_list()
    sparse_para=0.5
    EPS=1e-10
    lambda_multiply = Lambda(lambda x: tf.multiply(x[0], x[1]))
    lambda_batchdot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
        # l2 normalize
    lambda_l2nor = Lambda(lambda x: x / (tf.norm(x, ord=2, axis=-2, keepdims=True)+EPS))
    lambda_mean = Lambda(lambda x: K.mean(x, axis=-2,keepdims=True))
    lambda_l1nor = Lambda(lambda x: x / (tf.norm(x, ord=1, axis=-2, keepdims=True)+EPS))
    lambda_min = Lambda(lambda x: K.min(x, axis=-2,keepdims=True))
    
    layernorm_1= LayerNormalization()(input_tensor)
    QKV_Pre=Conv2D(filters*3, 1, padding='same',use_bias=False)(layernorm_1)
    QKV_F=DepthwiseConv2D(kernel_size=(3, 3), padding='same',use_bias=False)(QKV_Pre)  #b,w,h,filters*3
    
    Q_F,K_F,V_F=Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(QKV_F)
    
    Q_F_head=K.concatenate(Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': nhead})(Q_F), axis=0)
    K_F_head=K.concatenate(Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': nhead})(K_F), axis=0)
    V_F_head=K.concatenate(Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': nhead})(V_F), axis=0)
       
    Q_matrix=lambda_l2nor(Reshape((h * w, c//nhead))(Q_F_head))
    K_matrix_pemu=Permute((2, 1))(lambda_l2nor(Reshape((h * w, c//nhead))(K_F_head)))
    
    QK=lambda_batchdot([K_matrix_pemu, Q_matrix]) # (b*nhead)*filters//nhead * filters//nhead
    temperature =MyLayer(head=nhead)
    QK = K.concatenate(Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': nhead})(K.expand_dims(QK, axis=0)), axis=0)
    QK= K.squeeze(K.concatenate(Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': nhead})(temperature(QK)), axis=1), axis=0)
    
    QK_SP_softmax2=Softmax(axis=-2)(QK)
    QK_SP_Relu = Activation('relu')(QK)
    
  #  lambda_sum = Lambda(lambda x:K.sum(x,axis=-2))
  #  lambda_divide = Lambda(lambda x: tf.divide(x[0], x[1]))
    #QK_SP_Relu_sum = lambda_sum(QK_SP_Relu)+EPS
   # QK_SP_Relu_norm = lambda_divide([QK_SP_Relu,QK_SP_Relu_sum])
    QK_SP_Relu_norm = lambda_l1nor(QK_SP_Relu)
    
    at_active = lambda_multiply([QK_SP_softmax2, QK_SP_Relu_norm])  

    V_matrix=Reshape((h * w, c//nhead))(V_F_head)
    add_matrix=lambda_batchdot([V_matrix, at_active])
    add_F=Reshape((h , w, c//nhead))(add_matrix)
    add_F=K.concatenate(Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': nhead})(add_F), axis=3)
    
    add_linear=Conv2D(filters, 1, padding='same',use_bias=False)(add_F)    
    output=input_tensor+add_linear
    return output



def SCGFN(input_tensor, filters):
    [b, h, w, c] = input_tensor.get_shape().as_list()
    lambda_multiply = Lambda(lambda x: tf.multiply(x[0], x[1]))
    layernorm_1= LayerNormalization()(input_tensor)
    gamma=1
    F_Pre_3 = Conv2D(int(filters*gamma), 1, padding='same',use_bias=False)(layernorm_1)
    
    Spatial_F_gating_1= Conv2D(1, 1, padding='same',use_bias=False)(layernorm_1)
    Spatial_F_gating_2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same',use_bias=False)(Spatial_F_gating_1)
    Spatial_F_gating_3 = Activation('relu')(Spatial_F_gating_2)  
    Spa_F=lambda_multiply([F_Pre_3, Spatial_F_gating_3])
    
    Channel_F_gating_1 = Conv2D(int(filters*gamma), 1, padding='same',use_bias=False)(layernorm_1)
    Channel_F_gating_2 = GlobalAveragePooling2D()(Channel_F_gating_1)
    Channel_F_gating_2 = Reshape((1 , 1, int(filters*gamma)))(Channel_F_gating_2)
    Channel_F_gating_3 = Activation('relu')(Channel_F_gating_2)  
    Cha_F=lambda_multiply([F_Pre_3, Channel_F_gating_3])
    
    SC_F=K.concatenate([Spa_F, Cha_F], axis=3)
    add_linear=Conv2D(filters, 1, padding='same',use_bias=False)(SC_F)  
    output=input_tensor+add_linear
    return output  


def model_arch_cloud(input_rows=256, input_cols=256, num_of_channels=4, num_of_classes=1):
    inputs = Input((input_rows, input_cols, num_of_channels))
   
    dim_c=32 
    conv_CloudyO=Conv2D(dim_c, (3, 3),activation='relu',padding='same',use_bias=False)(inputs)
    conv_CloudyO=SDTA(conv_CloudyO, dim_c,1)
    conv_CloudyO=SCGFN(conv_CloudyO, dim_c)
    
    
    down1_O=Downs(conv_CloudyO) #dim_c*2
    down1_O=SDTA(down1_O, dim_c*2,2)
    down1_O=SCGFN(down1_O, dim_c*2)
    
    
    down2_O=Downs(down1_O) #dim_c*4
    down2_O=SDTA(down2_O, dim_c*4,4)
    down2_O=SCGFN(down2_O, dim_c*4)
    
    
    down3_O=Downs(down2_O) #dim_c*8
    down3_O=SDTA(down3_O, dim_c*8,8)
    down3_O=SCGFN(down3_O, dim_c*8)


    up_f_1=Ups(down3_O)
    up_f_1 = concatenate([up_f_1, down2_O], axis=3)
    up_f_1=Conv2D(dim_c*4, 1, padding='same',use_bias=False)(up_f_1)
    up_f_1_mdta=SDTA(up_f_1, dim_c*4,4)
    up_f_1_gdfn=SCGFN(up_f_1_mdta, dim_c*4)


    up_f_2=Ups(up_f_1_gdfn)
    up_f_2 = concatenate([up_f_2, down1_O], axis=3)
    up_f_2 = Conv2D(dim_c*2, 1, padding='same',use_bias=False)(up_f_2)
    up_f_2_mdta=SDTA(up_f_2, dim_c*2,2)
    up_f_2_gdfn=SCGFN(up_f_2_mdta, dim_c*2)


    up_f_3=Ups(up_f_2_gdfn)
    up_f_3 = concatenate([up_f_3, conv_CloudyO], axis=3)
    up_f_3_mdta=SDTA(up_f_3, dim_c*2,1)
    up_f_3_gdfn=SCGFN(up_f_3_mdta, dim_c*2)

    
    refine_mdta=SDTA(up_f_3_gdfn, dim_c*2,1)
    refine_gdfn=SCGFN(refine_mdta, dim_c*2)


    conv12 = Conv2D(6, (3, 3), activation=None, padding='same')(refine_gdfn )

    return Model(inputs=[inputs], outputs=[conv12])


