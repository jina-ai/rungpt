class QuantizeArgs:
    def __init__(
        self,
        model_name,
        model_path,
        dataset: str = 'wikitext2',
        load_from_saved: str = None,
        seed: int = 0,
        nsamples: int = 128,
        percdamp: float = 0.01,
        wbits: int = 4,
        groupsize: int = 16,
        permutation_order: str = "identity",
        true_sequential: bool = False,
        new_eval: bool = False,
        sym: bool = False,
        perchannel: bool = True,
        qq_scale_bits: int = 3,
        round_zero: int = None,
        qq_zero_bits: int = 3,
        qq_zero_sym: bool = False,
        qq_groupsize: int = 16,
        outlier_threshold: float = 0.2,
        simplified_outliers: bool = False,
        save: str = '',
        save_safetensors: str = '',
        benchmark: int = 0,
        check: bool = False,
        skip_out_loss: bool = False,
        offload_activations: bool = False,
        dtype: str = "auto",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.dataset = dataset
        self.load_from_saved = load_from_saved
        self.seed = seed
        self.nsamples = nsamples
        self.percdamp = percdamp
        self.wbits = wbits
        self.groupsize = groupsize
        self.permutation_order = permutation_order
        self.true_sequential = true_sequential
        self.new_eval = new_eval
        self.sym = sym
        self.perchannel = perchannel
        self.qq_scale_bits = qq_scale_bits
        self.round_zero = round_zero
        self.qq_zero_bits = qq_zero_bits
        self.qq_zero_sym = qq_zero_sym
        self.qq_groupsize = qq_groupsize
        self.outlier_threshold = outlier_threshold
        self.simplified_outliers = simplified_outliers
        self.save = save
        self.save_safetensors = save_safetensors
        self.benchmark = benchmark
        self.check = check
        self.skip_out_loss = skip_out_loss
        self.offload_activations = offload_activations
        self.dtype = dtype
