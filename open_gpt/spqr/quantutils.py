# Adapted from https://github.com/Vahe1994/SpQR
import time
import torch
import torch.nn as nn
from tqdm import trange
from transformers import LlamaForCausalLM, LlamaTokenizer
from spqr_engine import SPQRUtil


def save_llama(model_name, model, save_directory):
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    LlamaTokenizer.save_pretrained(tokenizer, save_directory=save_directory)
    LlamaForCausalLM.save_pretrained(model, save_directory=save_directory)


def get_llama(model_path):
    import torch
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # preserving
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto"
    )

    model.seqlen = 2048
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_average_number_of_bits(
        wbits: int = 3,
        qq_scale_bits: int = 3,
        qq_zero_bits: int = 3,
        qqq_scale_bits: int = 16,
        qqq_zero_bits: int = 16,
        groupsize: int = 16,
        qq_groupsize: int = 16,
        round_zero: bool = False,
        global_ol_n_share: float = 0.00,
):
    # if not quantized stats are in full precision
    qq_scale_bits = qq_scale_bits or 16
    qq_zero_bits = qq_zero_bits or 16
    groupsize = groupsize or float('inf')
    qq_groupsize = qq_groupsize or float('inf')

    if round_zero:
        wbits_avg = wbits + (qq_scale_bits + wbits) / groupsize + (qqq_scale_bits + qqq_zero_bits) / (
                groupsize * qq_groupsize)
    else:
        wbits_avg = wbits + (qq_scale_bits + qq_zero_bits) / groupsize + 2 * (qqq_scale_bits + qqq_zero_bits) / (
                groupsize * qq_groupsize)

    # correct accounting for outliers
    if global_ol_n_share > 0:
        wbits_avg += 32 * global_ol_n_share

    return round(wbits_avg, 2)


@torch.no_grad()
def llama_sequential(model, dataloader, args, dev):
    print("\nStarting SPQR compression ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in dataloader:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    quantizers = {}
    normal_outlier_count_global, w_count_global = 0, 0

    for i in range(len(layers)):
        print(f"\n------------------------------------------------------------------\nStarting layer {i}")
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}

        start_time = time.time()
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = SPQRUtil(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Quantizing module {name} of layer {i}")
                quantized = gptq[name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=args.qq_zero_sym,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                )

                gptq[name].layer.weight.data = quantized.weight.to(gptq[name].layer.weight.data.dtype)
                quantizers["model.layers.%d.%s" % (i, name)] = ()  # to be updated

                # OUTLIER STATS per module:
                normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()

                stats_payload[f"n_{name}_ol_share"] = round((normal_outliers_count / quantized.weight.numel()).item(),
                                                            6)

                normal_outlier_count += normal_outliers_count.item()
                w_count += quantized.weight.numel()

        # upload inputs back to the device
        if args.offload_activations:
            inps = inps.to(dev)
            outs = outs.to(dev)

        if not args.skip_out_loss:
            outs_tmp = outs.clone()

        for j in trange(args.nsamples, desc="applying", leave=False):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        if args.skip_out_loss:
            out_losses = torch.full((1,), torch.nan)
        else:
            out_losses = (outs - outs_tmp).float().square().view(
                outs.shape[0], -1
            ).mean(dim=1).sqrt() / outs.view(outs.shape[0], -1).float().std(dim=1)
            del outs_tmp

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["ol_share"] = round(normal_outlier_count / w_count, 6)
        stats_payload["out_loss"] = torch.mean(out_losses).item()
        stats_payload["Step"] = i

        normal_outlier_count_global += normal_outlier_count
        w_count_global += w_count

        print(stats_payload)

    print("=====================\nFinal stats:")
    print(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")

    wbits_avg = get_average_number_of_bits(
        args.wbits,
        args.qq_scale_bits,
        args.qq_zero_bits,
        16,
        16,
        args.groupsize,
        args.qq_groupsize,
        args.round_zero,
        normal_outlier_count_global / w_count_global
    )

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i, end=", ", flush=True)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\nperplexity = {ppl.item():.4f}")

    model.config.use_cache = use_cache
