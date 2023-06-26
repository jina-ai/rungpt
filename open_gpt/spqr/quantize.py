# Adapted from https://github.com/Vahe1994/SpQR
import time

from quantizeargs import QuantizeArgs
from datautils import *
from quantutils import *
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, args, device)
        print(time.time() - tick)

    if quantized_model_path is not None:
        save_llama(model, quantized_model_path)


def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_llama(args.model_path)

    if args.benchmark:
        raise NotImplementedError()

    datasets = ["wikitext2", "ptb", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model_path=args.model_path,
                                             seqlen=model.seqlen)
        print(dataset)
        args.dataset = dataset
        llama_eval(model, testloader, args, device)

    if args.save or args.save_safetensors:
        raise NotImplementedError()

def quant(model_name, quantized_model_path):
    model_path = huggingface_hub.snapshot_download(model_name)
    before_args = QuantizeArgs(model_path=model_path)
    test(before_args)
    apply_quantize(before_args, quantized_model_path)
    quantized_args = QuantizeArgs(model_path=quantized_model_path)
    test(quantized_args)
    return before_args, quantized_args

