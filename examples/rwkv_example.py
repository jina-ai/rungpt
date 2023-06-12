import open_gpt
from open_gpt.profile import end_measure, log_measures, start_measure

prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

start_measures = start_measure()
model = open_gpt.create_model(
    'ybelkada/rwkv-raven-1b5',
    precision='fp16',
    device_map='balanced'
    # 'sgugger/rwkv-430M-pile', precision='fp16', device_map='balanced'
    # 'sgugger/rwkv-7b-pile', precision='fp16', device_map='balanced'
)

generated_text = model.generate(
    prompt, max_new_tokens=256, do_sample=True, temperature=0.9
)
print(f'==> {prompt} {generated_text}')

embedings = model.embedding(prompt)
print(f'==> {prompt} \n({embedings.shape})->\n{embedings}')


end_measures = end_measure(start_measures)
log_measures(end_measures, "Model generation")
