import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'ViT-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    # config.hidden_size = 768
    config.hidden_size = 128
    # config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072
    config.transformer.mlp_dim = 64*2
    # config.transformer.mlp_dim = 512
    config.transformer.num_heads = 4
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    # config.classifier = 'token'
    config.classifier = 'None'
    config.representation_size = None
    # config.representation_size = [2048, 1024]
    return config

def get_origin_b16_config():
    """basically the config I used for training SAM."""
    config = ml_collections.ConfigDict()
    config.name = 'ViT-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    # config.hidden_size = 768
    config.hidden_size = 64
    # config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072
    config.transformer.mlp_dim = 64*2
    # config.transformer.mlp_dim = 512
    config.transformer.num_heads = 4
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    # config.classifier = 'token'
    config.classifier = 'None'
    config.representation_size = None
    # config.representation_size = [2048, 1024]
    return config



def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.name = 'ViT-B_32'
    config.patches.size = (32, 32)
    
    return config

def get_sam_config():
    config = ml_collections.ConfigDict()
    config.rho = 0.0
    config.gradient_clipping = 1.0
    config.weight_decay = 0.0

    return config