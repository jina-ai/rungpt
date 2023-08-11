import open_gpt
from open_gpt.profile import LLMMeasure, end_measure, log_measures, start_measure

PROMPT = 'The goal of life is'


def run_benchmark(model, max_new_tokens, llm_measure):
    for i in range(args.repeat_time):
        llm_measure.start_record()
        generated_text = model.generate(
            PROMPT, max_new_tokens=max_new_tokens, do_sample=args.do_sample
        )
        llm_measure.end_record(generated_text['choices'][0]['text'])
    llm_measure.stats(stage='prefill' if max_new_tokens == 1 else 'decoding')
    llm_measure.clear()


def main(args):
    llm_measure = LLMMeasure()

    print(f"===> start model loading ...")
    model_load_start = start_measure()
    if args.precision == 'fp16':
        model = open_gpt.create_model(
            args.model_name,
            precision='fp16',
            device_map=args.device_map,
            backend=args.backend,
        )
    else:
        model = open_gpt.create_model(
            args.model_name,
            precision=args.precision,
            adapter_name_or_path=args.adapter_name,
            device_map=args.device_map,
            backend=args.backend,
        )
    model_load_end = end_measure(model_load_start)
    log_measures(model_load_end, "Resource measure")

    # warm up
    for i in range(10):
        _ = model.generate(PROMPT, max_new_tokens=5, do_sample=args.do_sample)

    print(f"===> start benchmark for prefill ...")
    prefill_start = start_measure(clear_cache=False)
    run_benchmark(model, max_new_tokens=1, llm_measure=llm_measure)
    prefill_end = end_measure(prefill_start)
    log_measures(prefill_end, "Resource measure")

    print(f"===> start benchmark for decoding ...")
    decoder_start = start_measure(clear_cache=False)
    run_benchmark(model, max_new_tokens=args.max_new_tokens, llm_measure=llm_measure)
    decoder_end = end_measure(decoder_start)
    log_measures(decoder_end, "Resource measure")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark for open_gpt.')
    parser.add_argument(
        '--model-name',
        type=str,
        default='decapoda-research/llama-7b-hf',
        help='model name to perform benchmark',
    )
    parser.add_argument(
        '--adapter-name',
        type=str,
        default=None,
        help='adapter name to perform benchmark',
    )
    parser.add_argument(
        '--precision', type=str, default='fp16', help='precision used for inference'
    )
    parser.add_argument(
        '--device-map', type=str, default='balanced', help='device map for inference'
    )
    parser.add_argument(
        '--backend', type=str, default='hf', help='backend for inference'
    )
    parser.add_argument(
        '--repeat-time', type=int, default=100, help='repeat time for benchmark'
    )
    parser.add_argument(
        '--do-sample',
        type=bool,
        default=False,
        help='whether to use sampling for inference',
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=100, help='max new tokens for inference'
    )

    args = parser.parse_args()

    assert args.precision in [
        'fp16',
        'bit4',
        'bit8',
    ], 'precision must be fp16 or bit4 or bit8'

    main(args)
