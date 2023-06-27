# Adapted from https://github.com/Vahe1994/SpQR

import time
from quantizeargs import QuantizeArgs
from datautils import *
from quantutils import *
import huggingface_hub


def apply_quantize(args: QuantizeArgs, quantized_model_path: str = None):

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
    if args.wbits < 16:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, args, device)
        print(time.time() - tick)

    if quantized_model_path is not None:
        save_llama(args.model_name, model, quantized_model_path)


def test(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_llama(model_path)

    datasets = ["wikitext2", "ptb", "c4"]

    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, seed=0, model_path=model_path, seqlen=model.seqlen)
        print(dataset)
        llama_eval(model, testloader, device)


def quant(model_name, quantized_model_path):
    model_path = huggingface_hub.snapshot_download(model_name)
    before_args = QuantizeArgs(model_name=model_name, model_path=model_path)
    apply_quantize(before_args, quantized_model_path)
    quantized_args = QuantizeArgs(model_name=model_name,model_path=quantized_model_path)
    return before_args, quantized_args

