import tensorflow as tf
import tensorflow_addons as tfa

import net.resnet as resnet_layer

class Graph_Convolution(tf.keras.Model):
    def __init__(self, out_dim, use_bias,  name = None):
        super(Graph_Convolution, self).__init__(name)
        self.use_bias = use_bias
        self.out_dim = out_dim
        if use_bias:
            self.gcn_b = tf.Variable(
                lambda:tf.keras.initializers.Zeros()(shape = [out_dim]), trainable=True, name="gcn_b"
            )
        
    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]
    
    def call(self, input, adj):
        x = input
        # print("x.shape:", x.shape) #  (256, 197, in_dim)
        # print("adj.shape:", adj.shape) # (197, 197) 
        x = self.get("gcn_dense",tf.keras.layers.Dense, self.out_dim, use_bias=False)(x)
        x = tf.matmul(adj, x)
        # print("x.shape:", x.shape) # (256, 197, out_dim)

        if self.use_bias:
            x += self.gcn_b

        return x
        

 




class Add_Position_Embs(tf.keras.Model):
  """Adds (optionally learned) positional embeddings to the inputs.
  ###### warning ######
  I don't know why but if this function inherit tf.keras.layers.Layer, the self.pe will become none trainable.

  Attributes:
    posemb_init: positional embedding initializer.

    # 197 since add 1 class token
    tf.Variable 'ViT-B_16/encoder/posembed_input:0' shape=(1, 197, 64) dtype=float32
    embedding conn2d: (16, 16, 3, 64)
  """

  def __init__(self, input_shape, name = None):
    super(Add_Position_Embs, self).__init__(name)
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert len(input_shape) == 3, "tensor must rank 3"
    pos_emb_shape = (1, input_shape[1], input_shape[2])
    self.pe = tf.Variable(
            lambda:tf.keras.initializers.RandomNormal(stddev=0.02)(shape = pos_emb_shape), trainable=True, name="posembed_input"
        )

    
    
    

  def call(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """

    inputs = inputs + self.pe

    return inputs

class Mlp_Block(tf.keras.Model):
    """Transformer MLP / feed-forward block."""

    # mlp_dim: int
    # dtype: Dtype = jnp.float32
    # out_dim: Optional[int] = None
    # dropout_rate: float = 0.1
    # kernel_init: Callable[[PRNGKey, Shape, Dtype],
    #                         Array] = nn.initializers.xavier_uniform()
    # bias_init: Callable[[PRNGKey, Shape, Dtype],
    #                     Array] = nn.initializers.normal(stddev=1e-6)
    
    def __init__(self, mlp_dim,out_dim = None,dropout_rate = 0.1, name = None):
        super(Mlp_Block, self).__init__(name)
        # actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.d1 = tf.keras.layers.Dense(self.mlp_dim,bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6))
        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
        if out_dim is None:
            self.d2 = None
        else:
            self.d2 = tf.keras.layers.Dense(self.out_dim,bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6))
        self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, inputs, deterministic):
        """Applies Transformer MlpBlock module."""
        if self.d2 is None:
            self.d2 = tf.keras.layers.Dense(inputs.shape[-1],bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6))

        x = self.d1(inputs)
        x = tf.keras.activations.gelu(x,approximate = True)

        x = self.drop1(x, not deterministic)
        output = self.d2(x)

        output = self.drop2(output, not deterministic)
        return output

class Encoder_1D_Block(tf.keras.Model):
    """Transformer encoder layer.

    Attributes:
        inputs: input data.
        mlp_dim: dimension of the mlp on top of attention block.
        dtype: the dtype of the computation (default: float32).
        dropout_rate: dropout rate.
        attention_dropout_rate: dropout for attention heads.
        deterministic: bool, deterministic or not (to apply dropout).
        num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    # mlp_dim: int
    # num_heads: int
    # dtype: Dtype = jnp.float32
    # dropout_rate: float = 0.1
    # attention_dropout_rate: float = 0.1

    def __init__(self, mlp_dim,num_heads,dropout_rate = 0.1, attention_dropout_rate = 0.1, name = None):
        super(Encoder_1D_Block, self).__init__(name = name)
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.ln1 = tf.keras.layers.LayerNormalization(axis = [1,2],epsilon=1e-06,) # axis igonores 0, and we make sure the rank is 3.
        self.ln2 = tf.keras.layers.LayerNormalization(axis = [1,2],epsilon=1e-06,) # axis igonores 0, and we make sure the rank is 3.
        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.mlp_block = Mlp_Block(mlp_dim=self.mlp_dim,dropout_rate=self.dropout_rate)


    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]


    def call(self, inputs, deterministic):
        """Applies Encoder1DBlock module.

        Args:
        inputs: Inputs to the layer.
        deterministic: Dropout will not be applied when set to true.

        Returns:
        output after transformer encoder block.
        """

        # Attention block.
        assert len(inputs.shape) == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = self.ln1(inputs)
        if not deterministic:
            x = self.get("attn1",tf.keras.layers.MultiHeadAttention,self.num_heads,x.shape[-1],x.shape[-1], self.attention_dropout_rate,False)(x, x, x)
        else:
            attn1 = self.get("attn1",tf.keras.layers.MultiHeadAttention,self.num_heads,x.shape[-1],x.shape[-1], self.attention_dropout_rate,False)
            x = attn1(x, x, x,training = False)
        x = self.drop1(x, not deterministic)
        x = x + inputs

        # MLP block.
        y = self.ln2(x)
        y = self.mlp_block(y, deterministic=deterministic)

        return x + y

class Encoder(tf.keras.Model):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
        num_layers: number of layers
        mlp_dim: dimension of the mlp on top of attention block
        num_heads: Number of heads in nn.MultiHeadDotProductAttention
        dropout_rate: dropout rate.
        attention_dropout_rate: dropout rate in self attention.
    """

    # num_layers: int
    # mlp_dim: int
    # num_heads: int
    # dropout_rate: float = 0.1
    # attention_dropout_rate: float = 0.1

    def __init__(self, num_layers, mlp_dim,num_heads, dropout_rate = 0.1, attention_dropout_rate = 0.1, name = None):
        super(Encoder, self).__init__(name)
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate


        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)

        self.e_1d_blocks = [Encoder_1D_Block(mlp_dim=self.mlp_dim, 
                    num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,name = f'encoderblock_{lyr}',
                    ) for lyr in range(self.num_layers) ]
        self.ln1 = tf.keras.layers.LayerNormalization(axis = [1,2],epsilon=1e-06,) # axis igonores 0, and we make sure the rank is 3.

        '''
        nodes: 196 => 64 => 32 => 8
        '''

        self.gcn1 = Graph_Convolution(64,use_bias = False, name = "gcn1")
        self.gcn2 = Graph_Convolution(32,use_bias = False, name = "gcn2")
        self.gcn3 = Graph_Convolution(8,use_bias = False, name = "gcn3")

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]
    
    def normalizing_adj(self,adj):
        """Symmetrically normalize adjacency matrix.
        
        after normalize, it will be transpose!!
        """
        
        rowsum = tf.reduce_sum(adj, axis= 1)
        d_inv_sqrt = tf.math.pow(rowsum, -0.5)
        d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
        return tf.matmul(tf.transpose(tf.matmul(adj,d_mat_inv_sqrt), [1,0]), d_mat_inv_sqrt) 


    def diff_pool(self, input, gcn, adj):
        s_transform = gcn(input, adj) # (batch, old_clusters, max_clusters)
        s_transform = tf.keras.activations.softmax(s_transform, axis=2) #  # (batch, old_clusters, max_clusters)
        coarse_x = tf.matmul(s_transform, input, transpose_a=True)
        # print("coarse_x1:", coarse_x1.shape) # (batch, dim, old_clusters) *  (batch, old_clusters, max_clusters) =  (batch, max_clusters, dim)

        tmp_as = tf.matmul(adj, s_transform)
        # print("tmp_as:", tmp_as.shape) # (batch, old_clusters, max_clusters)
        new_adj = tf.matmul(s_transform, tmp_as, transpose_a=True)
        # print("new_adj:", new_adj.shape) # (batch, max_clusters, max_clusters)

        return coarse_x, new_adj



    def call(self, inputs, train):
        """Applies Transformer model on the inputs.

        Args:
        inputs: Inputs to the layer.
        train: Set to `True` when training.

        Returns:
        output of a transformer encoder.


        num_layers: 8
        encoder: 1,2,4
        decoder: 1

        """
        assert len(inputs.shape) == 3  # (batch, len, emb)


        # print("inputs_shape:",inputs) # (256, 197, 64


  
        # x = self.add_pe(inputs)
        x = self.get("add_pe", Add_Position_Embs, inputs.shape)(inputs)
        # x = inputs + self.get("posembed_input",tf.Variable, lambda:tf.keras.initializers.RandomNormal(stddev=0.02)(shape = inputs.shape),trainable=True)
        x = self.drop1(x, train)

        this_pe = self.weights[-1][0] # for get rid of batch dim
        normalized_pe = tf.math.l2_normalize(this_pe, -1)
        raw_adj = tf.linalg.matmul(this_pe, tf.transpose(this_pe,[1,0]))
        raw_adj = tf.math.abs(raw_adj) # |cos theta of vectors|
        normalized_adj = self.normalizing_adj(raw_adj)

        '''
        encoder
        '''
        print("x.shape:", x.shape)
        x = self.e_1d_blocks[0](x, deterministic=not train)
        x = self.e_1d_blocks[1](x, deterministic=not train)
        print("x.shape:", x.shape)
        x, new_adj1 = self.diff_pool(x, self.gcn1, normalized_adj)
        print("x.shape:", x.shape)
        x = self.e_1d_blocks[2](x, deterministic=not train)
        x = self.e_1d_blocks[3](x, deterministic=not train)
        print("x.shape:", x.shape)
        x, new_adj2 = self.diff_pool(x, self.gcn2, new_adj1)
        print("x.shape:", x.shape)
        x = self.e_1d_blocks[4](x, deterministic=not train)
        x = self.e_1d_blocks[5](x, deterministic=not train)
        print("x.shape:", x.shape)
        x, new_adj3 = self.diff_pool(x, self.gcn3, new_adj2)
        print("x.shape:", x.shape)
        x = self.e_1d_blocks[6](x, deterministic=not train)
        print("x.shape:", x.shape)

        '''
        decoder
        '''
        x = self.e_1d_blocks[7](x, deterministic=not train)
        encoded = self.ln1(x) ## need to check x rank!!!!

        return encoded



class Classical_Encoder(tf.keras.Model):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
        num_layers: number of layers
        mlp_dim: dimension of the mlp on top of attention block
        num_heads: Number of heads in nn.MultiHeadDotProductAttention
        dropout_rate: dropout rate.
        attention_dropout_rate: dropout rate in self attention.
    """

    # num_layers: int
    # mlp_dim: int
    # num_heads: int
    # dropout_rate: float = 0.1
    # attention_dropout_rate: float = 0.1

    def __init__(self, num_layers, mlp_dim,num_heads, dropout_rate = 0.1, attention_dropout_rate = 0.1, name = None):
        super(Classical_Encoder, self).__init__(name)
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate


        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)

        self.e_1d_blocks = [Encoder_1D_Block(mlp_dim=self.mlp_dim, 
                    num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,name = f'encoderblock_{lyr}',
                    ) for lyr in range(self.num_layers) ]
        self.ln1 = tf.keras.layers.LayerNormalization(axis = [1,2],epsilon=1e-06,) # axis igonores 0, and we make sure the rank is 3.

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]


    def call(self, inputs, train):
        """Applies Transformer model on the inputs.

        Args:
        inputs: Inputs to the layer.
        train: Set to `True` when training.

        Returns:
        output of a transformer encoder.
        """
        assert len(inputs.shape) == 3  # (batch, len, emb)


  
        # x = self.add_pe(inputs)
        x = self.get("add_pe", Add_Position_Embs, inputs.shape)(inputs)
        # x = inputs + self.get("posembed_input",tf.Variable, lambda:tf.keras.initializers.RandomNormal(stddev=0.02)(shape = inputs.shape),trainable=True)
        x = self.drop1(x, train)

        # Input Encoder
        for e_lyr in self.e_1d_blocks:
            x = e_lyr(x, deterministic=not train)
        encoded = self.ln1(x) ## need to check x rank!!!!

        return encoded

class ViT(tf.keras.Model):
    """VisionTransformer."""

    # `num_classes: int
    # patches: Any
    # patches: Any
    # transformer: Any
    # hidden_size: int
    # resnet: Optional[Any] = None
    # representation_size: Optional[int] = None
    # classifier: str = 'token'

    def __init__(self, num_classes, mode , patches, transformer, hidden_size, resnet = None, representation_size = None, classifier = "token", name = "VIT"):
        super(ViT, self).__init__(name = name)
        self.num_classes = num_classes
        self.mode = mode
        self.patches = patches
        self.transformer = transformer
        self.hidden_size =  hidden_size
        self.resnet = resnet
        self.representation_size = representation_size
        self.classifier = classifier

        cls_init = tf.keras.initializers.Zeros()(
            shape=[1,1,self.hidden_size]
        )

        self.cls = tf.Variable(cls_init,trainable = True,name = "cls")

        # self.head = tf.keras.layers.Dense(
        #     self.num_classes, kernel_initializer=tf.keras.initializers.LecunNormal(), name = "head")

        self.flat1 = tf.keras.layers.Flatten()
        self.head = tf.keras.layers.Dense(
            112*112*3, kernel_initializer=tf.keras.initializers.LecunNormal(), name = "head")
    
        self.up1 = tf.keras.layers.UpSampling2D(size = [2,2])

        self.origin_transform = tf.keras.layers.Dense(1024)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]


    def call(self, inputs, train = True):


        x = inputs
        # (Possibly partial) ResNet root.

        n, h, w, c = x.shape

        '''
        get patch here!!!
        '''
        # We can merge s2d+emb into a single conv; it's the same.
        x = self.get("embedding",tf.keras.layers.Conv2D, self.hidden_size, self.patches.size, self.patches.size, 
                padding = "valid",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)

        # Here, x is a grid of embeddings.

        # Transformer.
        n, h, w, c = x.shape
        x = tf.reshape(x, [-1, h * w, c])

        # If we want to add a class token, add it here.
        if self.classifier == 'token':
            cls = self.cls
            # cls = tf.tile(cls,[1, 1, 1])
            inter_cls = tf.zeros_like(x[:,0:1,:]) # (b, 1, 32)
            inter_cls = inter_cls + cls #() (b, 1, 32)
            cls = inter_cls
            # print("cls_tile:",cls.shape) # (b, 1, 32)
            # print("x.shape:", x.shape) # (2, 144, 32)
            x = tf.concat([cls, x], axis=1)

        if self.mode == "origin":
            x = self.get('Transformer',Classical_Encoder, **self.transformer)(x, train=train)

        else:
            x = self.get('Transformer',Encoder, **self.transformer)(x, train=train)
        



        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = tf.reduce_mean(x, axis=list(range(1, len(x.shape) - 1)))  # (1,) or (1,2)
        else:
            x = self.flat1(x)
            print("flat_x:", x.shape)
            '''
            origin: (256, 12544)
            pooling: (256, 1024)
            '''


        if self.mode == "origin":
            x = self.origin_transform(x)

        if self.representation_size is not None:
            x = self.get("pre_logits",tf.keras.layers.Dense, 
                self.representation_size[0],kernel_initializer=tf.keras.initializers.LecunNormal(),
                )(x)
            # x = tf.keras.activations.gelu(x,approximate = True)
            x = self.get("pre_logits2",tf.keras.layers.Dense, 
                self.representation_size[1],kernel_initializer=tf.keras.initializers.LecunNormal(),
                )(x)
        # else:
        #     x = IdentityLayer(name='pre_logits')(x)

        '''
        the original implementation says using -10 for bias init(says in issue), which scale is too large in my view.
        '''
        # logit = self.get("head",tf.keras.layers.Dense, 
        #     self.num_classes, kernel_initializer=tf.keras.initializers.LecunNormal())(x) 

        logit = self.head(x)
        logit = tf.reshape(logit, [-1, 112, 112, 3])
        logit = self.up1(logit)

        return logit
        # prob = self.get("label",tf.keras.layers.Dense, 
        #     self.num_classes, kernel_initializer=tf.keras.initializers.LecunNormal(),activation = "softmax")(x) 
               
        # return prob
        