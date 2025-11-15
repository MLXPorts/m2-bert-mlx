from transformers import BertConfig


class BertConfig(BertConfig):

    def __init__(
        self,
        alibi_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,

        # mlp
        use_glu_mlp: bool = True,
        use_monarch_mlp: bool = False,
        monarch_mlp_nblocks: int = 4,

        # position
        use_positional_encodings: bool = False,
        max_position_embeddings: int = 512,

        # architecture selection
        monarch_mixer_sequence_mixing: bool = False,
        residual_long_conv: bool = False,
        
        # hyena and long conv hyperparameters
        long_conv_l_max: int = 128,
        long_conv_kernel_learning_rate: float = None,
        bidirectional: bool = True,
        hyena_lr_pos_emb: float = 1e-5,
        hyena_w: int = 10,
        hyena_w_mod: int = 1,
        hyena_wd: float = 0.1,
        hyena_emb_dim: int = 3,
        hyena_filter_dropout: float = 0.2,
        hyena_filter_order: int = 64,
        hyena_training_additions: bool = False,
        
        # efficiency
        use_flash_mm: bool = False,
        use_flash_fft: bool = False,

        # average pooling instead of CLS token
        pool_all: bool = False,
        use_cls_token: bool = True,

        # additional options
        sequence_token_planting: bool = False,
        attention_pooling: bool = False,
        gather_sentence_embeddings: bool = False,
        use_normalized_embeddings: bool = False,
        expand_positional_embeddings: bool = False,
        performing_BEIR_evaluation: bool = False,

        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size

        # mlp
        self.use_glu_mlp = use_glu_mlp
        self.use_monarch_mlp = use_monarch_mlp
        self.monarch_mlp_nblocks = monarch_mlp_nblocks

        # positional encodings
        self.use_positional_encodings = use_positional_encodings
        self.max_position_embeddings = max_position_embeddings

        # architecture
        self.monarch_mixer_sequence_mixing = monarch_mixer_sequence_mixing
        self.residual_long_conv = residual_long_conv

        # hyena and long conv hyperparameters
        self.long_conv_l_max = long_conv_l_max
        self.long_conv_kernel_learning_rate = long_conv_kernel_learning_rate
        self.bidirectional = bidirectional
        self.hyena_lr_pos_emb = hyena_lr_pos_emb
        self.hyena_w = hyena_w
        self.hyena_w_mod = hyena_w_mod
        self.hyena_wd = hyena_wd
        self.hyena_emb_dim = hyena_emb_dim
        self.hyena_filter_dropout = hyena_filter_dropout
        self.hyena_filter_order = hyena_filter_order
        self.hyena_training_additions = hyena_training_additions

        # efficiency
        self.use_flash_mm = use_flash_mm
        self.use_flash_fft = use_flash_fft

        # average pooling instead of CLS token
        self.pool_all = pool_all
        self.use_cls_token = use_cls_token

        # additional options
        self.sequence_token_planting = sequence_token_planting
        self.attention_pooling = attention_pooling
        self.gather_sentence_embeddings = gather_sentence_embeddings
        self.use_normalized_embeddings = use_normalized_embeddings
        self.expand_positional_embeddings = expand_positional_embeddings
        self.performing_BEIR_evaluation = performing_BEIR_evaluation

