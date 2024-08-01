import tensorflow as tf
class transformer():
    def __init__(self,batch_size = 3,image_shape = (304,608,1),out_size = (304,608),grid_shape = (3,304*608) , name='SpatialTransformer2dAffine') -> None:
        self.image_shape = image_shape
        self.height = image_shape[0]
        self.width = image_shape[1]
        self.channels = image_shape[2]
        self.affine_shape = 6
        self.grid_shape = grid_shape
        self.batch =batch_size
        self.out_size = out_size
    def interpolate(self,im,x,y):
        # constants
        num_batch = self.batch
        height = self.height
        width = self.width
        channels = self.channels
        out_size = self.out_size

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(height - 1, 'int32')
        max_x = tf.cast(width - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = tf.repeat(tf.range(num_batch) * dim1,out_height * out_width,axis = 0)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([height*width*num_batch, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output
    def __call__(self,data):
        # input shape ( bacth,h,w,c)
        # affines shape = (batch,2,3)
        # grid shape = (batch,3,h*w)
        
        """
        inputs = data[:,:tf.math.reduce_prod(self.image_shape)]
        inputs = tf.reshape(inputs,(self.batch,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        affines = data[:,tf.math.reduce_prod(self.image_shape):tf.math.reduce_prod(self.image_shape)+self.affine_shape]
        affines = tf.reshape(affines,shape = (self.batch,2,3))
        grid = data[:,tf.math.reduce_prod(self.image_shape)+self.affine_shape:]
        grid = tf.reshape(grid,(self.batch,self.grid_shape[0],self.grid_shape[1]))
        """
        inputs = data[0]
        affines = data[1]
        grid = data[2]

    
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(affines, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [self.batch, 1, self.grid_shape[1]])
        y_s = tf.slice(T_g, [0, 1, 0], [self.batch, 1, self.grid_shape[1]])
        x_s_flat = tf.reshape(x_s, [self.batch*self.grid_shape[1]])
        y_s_flat = tf.reshape(y_s, [self.batch*self.grid_shape[1]])

        input_transformed = self.interpolate(inputs, x_s_flat, y_s_flat)

        output = tf.reshape(input_transformed, tf.stack([self.batch, self.image_shape[0], self.image_shape[1], self.image_shape[2]]))
        return output
class Af(tf.keras.layers.Layer): 
    def __init__(self,bacth_size :int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.b_size = bacth_size
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    def build(self,input_shape):
        img_shape = input_shape[0]
        affine_shape = input_shape[1]
        self.img_shape = img_shape
        self.affine_shape = affine_shape
        x_t = tf.linspace(-1,1,img_shape[3])
        y_t = tf.linspace(-1,1,img_shape[2])
        x_t,y_t = tf.meshgrid(x_t,y_t)
        x_t = tf.reshape(x_t,(-1))
        y_t = tf.reshape(y_t,(-1))
        x_t = tf.cast(x_t,dtype= tf.float32)
        y_t = tf.cast(y_t,dtype= tf.float32)
        ones = tf.ones(tf.shape(x_t)[0])
        grid = tf.stack([x_t,y_t,ones],axis = 0)
        grid  = tf.expand_dims(grid,0)
        grid = tf.repeat(grid,img_shape[1]*self.b_size,axis = 0 )
        self.t = transformer(batch_size=img_shape[1]*self.b_size,
                             image_shape=img_shape[2:],
                             out_size=img_shape[2:-1],
                             grid_shape=(3,img_shape[2]*img_shape[3]))
        self.grid = grid
    def call(self,input):
        stimuli_input =input[0]
        eye_input = input[1]
        shape = tf.shape(stimuli_input)
        eye_shape = tf.shape(eye_input)
        stimuli_input = tf.cast(stimuli_input,dtype=tf.float32)
        stimuli_input = tf.reshape(stimuli_input,(-1,self.img_shape[2],self.img_shape[3],self.img_shape[4]))
        eye_input = tf.reshape(eye_input,(-1,2,3))
        eye_input = tf.cast(eye_input,dtype = tf.float32)

        out = self.t.__call__([stimuli_input,eye_input,self.grid])
        out = tf.reshape(out,shape)
        return out 

