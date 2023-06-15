import time

from quantizeargs import QuantizeArgs
from datautils import *
from main import *
import huggingface_hub


def apply_quantize(args: QuantizeArgs, quantized_model_path: str = None):
    if args.load:
        raise NotImplementedError()
    else:
        model = get_llama(args.model_path).train(False)

    if args.load_from_saved:
        dataloader = torch.load(args.load_from_saved)[: args.nsamples]
        testloader = None
    else:
        assert args.dataset != "custom"
        dataloader, testloader = get_loaders(
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
        quantizers = llama_sequential(model, dataloader, args, device)
        print(time.time() - tick)

    if quantized_model_path is not None:
        save_llama(model, quantized_model_path)
        return QuantizeArgs(model_path=quantized_model_path, dataset=args.dataset)


def test(before_args: QuantizeArgs, quantized_args: QuantizeArgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("before quantization:")
    model = get_llama(before_args.model_path)

    if before_args.benchmark:
        raise NotImplementedError()

    datasets = ["wikitext2", "ptb", "c4"]
    if before_args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, seed=before_args.seed, model_path=before_args.model_path,
                                             seqlen=model.seqlen)
        print(dataset)
        before_args.dataset_name = dataset
        llama_eval(model, testloader, before_args, device)

    if before_args.save or before_args.save_safetensors:
        raise NotImplementedError()

    print("after quantization:")
    model = get_llama(quantized_args.model_path)

    if quantized_args.benchmark:
        raise NotImplementedError()

    datasets = ["wikitext2", "ptb", "c4"]
    if quantized_args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, seed=quantized_args.seed, model_path=quantized_args.model_path,
                                             seqlen=model.seqlen)
        print(dataset)
        quantized_args.dataset_name = dataset
        llama_eval(model, testloader, quantized_args, device)

    if quantized_args.save or quantized_args.save_safetensors:
        raise NotImplementedError()


def quant(model_name, dataset, quantized_model_path):
    model_path = huggingface_hub.snapshot_download(model_name)
    before_args = QuantizeArgs(model_path=model_path,
                               dataset=dataset)
    quantized_args = apply_quantize(before_args, quantized_model_path)
    
    return before_args, quantized_args


before_args, quantized_args = quant("openlm-research/open_llama_3b", "wikitext2", "./quantized")
test(before_args, quantized_args)
