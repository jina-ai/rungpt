from cleo.commands.command import Command
from cleo.helpers import argument, option
import torch


class QuantizeArgs():
    def __init__(self,
                 model_path,
                 dataset: str = 'custom',
                 load_from_saved: str = None,
                 seed: int = 0,
                 nsamples: int = 128,
                 percdamp: float = 0.01,
                 nearest: bool = False,
                 wbits: int = 16,
                 groupsize: int = None,
                 permutation_order: str = "identity",
                 true_sequential: bool = False,
                 new_eval: bool = False,
                 sym: bool = False,
                 perchannel: bool = False,
                 qq_scale_bits: int = None,
                 round_zero: int = None,
                 qq_zero_bits: int = None,
                 qq_zero_sym: bool = False,
                 qq_groupsize: int = 16,
                 outlier_threshold: float = float("inf"),
                 simplified_outliers: bool = False,
                 save: str = '',
                 save_safetensors: str = '',
                 load: str = '',
                 benchmark: int = 0,
                 check: bool = False,
                 wandb: bool = False,
                 wandb_dir: str = '',
                 wandb_exp_name: str = 'SpQR',
                 skip_out_loss: bool = False,
                 offload_activations: bool = False,
                 dtype: str = "auto"):
        self.model_path = model_path
        self.dataset = dataset
        self.load_from_saved = load_from_saved,
        self.seed = seed,
        self.nsamples = nsamples,
        self.percdamp = percdamp,
        self.nearest = nearest,
        self.wbits = wbits,
        self.groupsize = groupsize,
        self.permutation_order = permutation_order,
        self.true_sequential = true_sequential,
        self.new_eval = new_eval,
        self.sym = sym,
        self.perchannel = perchannel,
        self.qq_scale_bits = qq_scale_bits,
        self.round_zero = round_zero,
        self.qq_zero_bits = qq_zero_bits,
        self.qq_zero_sym = qq_zero_sym,
        self.qq_groupsize = qq_groupsize,
        self.outlier_threshold = outlier_threshold,
        self.simplified_outliers = simplified_outliers,
        self.save = save,
        self.save_safetensors = save_safetensors,
        self.load = load,
        self.benchmark = benchmark,
        self.check = check,
        self.wandb = wandb,
        self.wandb_dir = wandb_dir,
        self.wandb_exp_name = wandb_exp_name,
        self.skip_out_loss = skip_out_loss,
        self.offload_activations = offload_activations,
        self.dtype = dtype


class QuantizeCommand(Command):
    name = "quantize"

    description = "quantize the model."

    arguments = [argument("model_path", "The path of the model to serve."),
                 argument("dataset", "The dataset for calibration.", default="none")]

    options = [
        option(
            'load_from_saved',
            None,
            'Path to load if specified.',
            flag=False,
            default=None,
        ),
        option(
            "percdamp",
            None,
            "Percent of the average Hessian diagonal to use for dampening.",
            flag=False,
            default=0.01,
        ),
        option(
            "wbits",
            None,
            "#bits to use for quantization; use 16 for evaluating base model.",
            flag=False,
            default=16,
        ),
        option(
            "groupsize",
            None,
            "How many weight columns (input features), are quantized with the same statistics, default = all of them.",
            flag=False,
            default=None,
        ),
        option(
            "permutation_order",
            None,
            "Weights permutation order; options: identity(default), spearman, act_order.",
            flag=False,
            default="identity",
        ),
        option(
            "true-sequential",
            None,
            "Whether to run in true sequential model.",
            flag=True,
        ),
        option(
            "new_eval",
            None,
            "if this is set, evaluate on new (and slightly more realistic!), val dataset versions.",
            flag=True,
        ),
        option(
            "sym",
            None,
            "Symmetric quantization.",
            flag=True,
        ),
        option(
            "perchannel",
            None,
            "fit a unique quantizer to each output dim.",
            flag=True,
        ),
        option(
            "qq_scale_bits",
            None,
            "Quantize quantization scale with this many bits (default=do not quantize).",
            default=None,
            flag=False,
        ),
        option(
            "round_zero",
            None,
            'whether to allow non-integer "zero" when quantizing weights non-symmetrically.',
            default=None,
            flag=False,
        ),
        option(
            "qq_zero_bits",
            None,
            "Quantize quantization \"zero\" with this many bits (default=do not quantize).",
            default=None,
            flag=False,
        ),
        option(
            "qq_zero_sym",
            None,
            "enable sym=True in meta-quantization for groupwise zero, specifically.",
            flag=True,
        ),
        option(
            "qq_groupsize",
            None,
            "Quantize quantization scale in groups of this many scales.",
            default=16,
            flag=False
        ),
        option(
            "outlier_threshold",
            None,
            "relative threshold for outliers; higher threshold = more outliers.",
            default=float("inf"),
            flag=False,
        ),
        option(
            "simplified_outliers",
            None,
            "do not perform leave-one-out evaluation when detecting outliers; works faster, but generally worse in perplexity.",
            flag=True,
        ),
        option(
            "skip_out_loss",
            None,
            "Whether to skip computation of out loss.",
            flag=True
        ),
        option(
            "offload_activations",
            None,
            "Whether to skip computation of out loss.",
            flag=True,
        )
    ]

    def handle(self) -> int:
        from SpQR import main, datautils
        import wandb, time

        args = QuantizeArgs(model_path=self.argument('model_name'),
                            dataset=self.argument('dataset'),
                            seed=self.argument('seed'),
                            nsamples=self.option('nsamples'),
                            percdamp=self.option('percdamp'),
                            nearest=self.option('nearest'),
                            wbits=self.option('wbits'),
                            groupsize=self.option('groupsize'),
                            permutation_order=self.option('permutation_order'),
                            true_sequential=self.option('true_sequential'),
                            new_eval=self.option('new_eval'),
                            sym=self.option('sym'),
                            perchannel=self.option('perchannel'),
                            qq_scale_bits=self.option('qq_scale_bits'),
                            round_zero=self.option('round_zero'),
                            qq_zero_bits=self.option('qq_zero_bits'),
                            qq_zero_sym=self.option('qq_zero_sym'),
                            qq_groupsize=self.option('qq_groupsize'),
                            outlier_threshold=self.option('outlier_threshold'),
                            simplified_outliers=self.option('simplified_outliers'),
                            save=self.option('save'),
                            save_safetensors=self.option('save_safetensors'),
                            load=self.option('load'),
                            benchmark=self.option('benchmark'),
                            check=self.option('check'),
                            wandb=self.option('wandb'),
                            wandb_dir=self.option('wandb_dir'),
                            wandb_exp_name=self.option('wandb_exp_name'),
                            skip_out_loss=self.option('skip_out_loss'),
                            offload_activations=self.option('offload_activations'),
                            dtype=self.option('dtype'))

        if type(args.load) is not str:
            args.load = args.load.as_posix()

        if args.load:
            raise NotImplementedError()
        else:
            model = main.get_llama(args.model_path).train(False)

        if args.load_from_saved:
            dataloader = torch.load(args.load_from_saved)[: args.nsamples]
            testloader = None
        else:
            assert args.dataset != "custom"
            dataloader, testloader = datautils.get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model_path=args.model_path, seqlen=model.seqlen
            )

        if args.wandb:
            args.exp_name = (
                    args.wandb_exp_name
                    + "_wbits_"
                    + str(args.wbits)
                    + "_groupsize_"
                    + str(args.groupsize)
                    + "_qq_scale_bits_"
                    + str(args.qq_scale_bits)
                    + "_qq_zero_bits_"
                    + str(args.qq_zero_bits)
                    + "_qq_groupsize_"
                    + str(args.qq_groupsize)
                    + "_outl_"
                    + str(args.outlier_threshold)
                    + "_permord_"
                    + str(args.permutation_order)
            )
            neweval_str = ""
            if args.new_eval:
                neweval_str = "_new_eval"
            wandb.init(
                name=args.exp_name,
                dir=args.wandb_dir,
                config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
            )
            wandb.run.log_code(".")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.load and args.wbits < 16 and not args.nearest:
            tick = time.time()
            quantizers = main.llama_sequential(model, dataloader, args, device)
            print(time.time() - tick)

        if args.benchmark:
            raise NotImplementedError()

        datasets = ["wikitext2", "ptb", "c4"]
        if args.new_eval:
            datasets = ["wikitext2", "ptb-new", "c4-new"]
        for dataset in datasets:
            dataloader, testloader = datautils.get_loaders(dataset, seed=args.seed, model_path=args.model_path,
                                                           seqlen=model.seqlen)
            print(dataset)
            args.dataset_name = dataset
            main.llama_eval(model, testloader, args, device)

        if args.save or args.save_safetensors:
            raise NotImplementedError()
