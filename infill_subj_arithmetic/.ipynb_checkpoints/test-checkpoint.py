import random
import json
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model

from utils import score_fast, remove_eos_and_pad_left, \
                append_sol_and_remove_eos

bsz = 32
grad_acc = 8
log_interval = 10

total_steps = 1000
warmup_m_steps = 20
total_m_steps = 100

lr = 0.0001

max_len = 5
reward_temp = 1

train_samples = 20
preseed_buffer = True

rngseed = 3

model_to_use = 'instruct-gpt-j-fp16' # 'gpt2'

if model_to_use == 'instruct-gpt-j-fp16':
    tokenizer = AutoTokenizer.from_pretrained('nlpcloud/instruct-gpt-j-fp16')
    model = AutoModelForCausalLM.from_pretrained('nlpcloud/instruct-gpt-j-fp16',
                                                torch_dtype=torch.bfloat16)
elif model_to_use == 'gpt2':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')  

model.to('cuda')

np.random.seed(0)
random.seed(0)

answers = [ 'objective', 'subjective' ]

obj_id = tokenizer.vocab['Ġobjective']
subj_id = tokenizer.vocab['Ġsubjective']

data_train = [ json.loads(l) for l in open('data/subj/train.jsonl', 'r') ]
data_test = [ json.loads(l) for l in open('data/subj/test.jsonl', 'r') ]

data_train = [sample for sample in data_train if len(sample['text'].split()) < 25]
data_test = [sample for sample in data_test]

random.shuffle(data_train)
data_train = data_train[:train_samples]

train_queries = []
train_sols = []

test_queries = []
test_sols = []

intro_prompt = 'Classify this movie review as objective or subjective: "'
cot_prompt = '" This review is'
sol_prompt = ', so it is'

for sample in data_train:
    train_queries.append(intro_prompt + sample['text'] + cot_prompt)
    train_sols.append(sol_prompt + ' ' + sample['label_text'] + '.')

for sample in data_test:
    test_queries.append(intro_prompt + sample['text'] + cot_prompt)
    test_sols.append(sol_prompt + ' ' + sample['label_text'] + '.')

n = 200
train_jsons = [json.dumps(x) for x in data_train][:n]
with open(f'data/subj/train.{n}.jsonl', 'w') as f:
    f.write('\n'.join(train_jsons))

encoded_train_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_queries]
encoded_train_sols = [tokenizer(answer, return_tensors='pt')['input_ids'].cuda() for answer in train_sols]
encoded_train_all_sols = [tokenizer(sol_prompt+' objective.', return_tensors='pt')['input_ids'].cuda(),
                          tokenizer(sol_prompt+' subjective.', return_tensors='pt')['input_ids'].cuda()]
encoded_test_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_queries]
encoded_sol_prompt = tokenizer(sol_prompt, return_tensors='pt')['input_ids'].cuda()

eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=256,
    lora_alpha=16,
    target_modules=["k_proj", "v_proj"] if model_to_use == 'instruct-gpt-j-fp16' else ["c_attn"],
    lora_dropout=0.,
    bias="none",
    modules_to_save=["classifier"],
)
knowledge_model = get_peft_model(model, lora_config)

def get_preds(model, encoded_queries, top_n = 999999, bsz = 1):
    preds = []
    encoded_obj = tokenizer(', so it is objective',
                                return_tensors='pt').to('cuda')['input_ids'][0]
    encoded_sub = tokenizer(', so it is subjective',
                                return_tensors='pt').to('cuda')['input_ids'][0]
    encoded_results = torch.nn.utils.rnn.pad_sequence([encoded_obj, encoded_sub], batch_first=True, padding_value=eos_token_id)
    encoded_queries_to_use = encoded_queries[:top_n]
    for i in range(len(encoded_queries_to_use) // bsz):
        batch_input = torch.nn.utils.rnn.pad_sequence([x[0] for x in encoded_queries_to_use[i*bsz:(i+1)*bsz]],
                                                      batch_first=True,
                                                      padding_value=eos_token_id)
        with torch.no_grad():
            mean_reward = score_fast(model,
                            append_sol_and_remove_eos(batch_input.repeat_interleave(2, dim=0),
                                                      encoded_results.repeat(bsz, 1), eos_token_id, pad_token_id),
                            eos_token_id=eos_token_id)
        pred = mean_reward.reshape(bsz, 2)
        preds += (pred[:, 0] > pred[:, 1]).tolist()
    return preds

breakpoint()

# M step
save_dir = '<YOUR SAVE DIR>'
ckpt_name = f'subj_obj_{model_to_use}_{train_samples}samples_len{max_len}_{total_steps}steps_rewtemp{reward_temp}_seed_{preseed_buffer}_rngseed_{rngseed}'

encoded_train_queries_w_cot_sample = torch.load(f'{save_dir}/{ckpt_name}/encoded_train_queries_w_cot_sample.pt')
encoded_test_queries_w_cot_greedy = torch.load(f'{save_dir}/{ckpt_name}/encoded_test_queries_w_cot_greedy.pt')
encoded_test_queries_w_cot_sample = torch.load(f'{save_dir}/{ckpt_name}/encoded_test_queries_w_cot_sample.pt')


true_preds = torch.tensor([True if 'objective' in sol else False for sol in test_sols])

knowledge_model.eval()

# E-step-only greedy
test_preds_greedy = get_preds(knowledge_model, encoded_test_queries_w_cot_greedy, bsz = 100)
greedy_preds = torch.tensor(test_preds_greedy)
print(f'Test Acc (greedy) : {(greedy_preds == true_preds).sum() / len(true_preds)}')
# E-step-only sample
test_preds_sample = get_preds(knowledge_model, [s.unsqueeze(0) for x in encoded_test_queries_w_cot_sample for s in x], bsz = 100)
shaped_preds = torch.tensor(test_preds_sample).reshape(len(encoded_test_queries_w_cot_sample), -1)
agg_preds = shaped_preds.sum(-1) / shaped_preds.size(1) > 0.5
print(f'Test Acc (sample) : {(agg_preds == true_preds).sum() / len(true_preds)}')

opt_knowledge = torch.optim.AdamW([{'params': knowledge_model.parameters(), 'lr': lr}], betas=(0.9, 0.99))

# learning rate schedule
def get_lr_mult_at_step(step):
    if step <= warmup_m_steps:
        return min(step/warmup_m_steps, 1.)
    return max((total_m_steps - step) / (total_m_steps - warmup_m_steps), 0)
sched = torch.optim.lr_scheduler.LambdaLR(opt_knowledge, get_lr_mult_at_step)

knowledge_model.train()
for step in range(total_m_steps):
    opt_knowledge.zero_grad()
    loss = 0.
    for _ in range(grad_acc):
        # build a batch
        batch_input = []
        batch_labels = []
        for _ in range(bsz):
            query_ind = np.random.choice(np.arange(len(encoded_train_queries_w_cot_sample)))
            rationale_ind = np.random.choice(np.arange(encoded_train_queries_w_cot_sample[query_ind].size(0)))
            encoded_input = encoded_train_queries_w_cot_sample[query_ind][rationale_ind]
            encoded_input = encoded_input[encoded_input != eos_token_id]
            encoded_input = append_sol_and_remove_eos(encoded_input.unsqueeze(0),
                                                      encoded_sol_prompt,
                                                      eos_token_id=eos_token_id,
                                                      pad_token_id=eos_token_id)[0]
            batch_input.append(encoded_input) # reverse to prepare for left-padding
            if 'objective' in train_sols[query_ind]:
                batch_labels.append(True)
            elif 'subjective' in train_sols[query_ind]:
                batch_labels.append(False)
        batch_input, position_ids, _ = \
            remove_eos_and_pad_left(batch_input, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
        position_ids = position_ids.cuda()
        batch_labels = torch.tensor(batch_labels, device='cuda', dtype=torch.bool)

        last_logprob = knowledge_model(batch_input,
                                       attention_mask=batch_input!=eos_token_id,
                                       position_ids=position_ids)['logits'][:, -1].log_softmax(dim=-1)
        obj_logprob = last_logprob[:, obj_id]
        subj_logprob = last_logprob[:, subj_id]
        partition_fn = torch.logsumexp(torch.stack([obj_logprob, subj_logprob], dim=-1), dim=-1)
        loss = torch.where(batch_labels, -(obj_logprob - partition_fn), -(subj_logprob - partition_fn))
        loss.mean().backward()

    opt_knowledge.step()
    sched.step()
    if step % log_interval == 0:
        print(f'loss: {loss.mean().item()}')

knowledge_model.eval()
# one-step EM greedy
test_preds_greedy = get_preds(knowledge_model, encoded_test_queries_w_cot_greedy, bsz = 100)
greedy_preds = torch.tensor(test_preds_greedy)
print(f'Test Acc (greedy) : {(greedy_preds == true_preds).sum() / len(true_preds)}')
# one-step EM sample
test_preds_sample = get_preds(knowledge_model, [s.unsqueeze(0) for x in encoded_test_queries_w_cot_sample for s in x], bsz = 100)
shaped_preds = torch.tensor(test_preds_sample).reshape(len(encoded_test_queries_w_cot_sample), -1)
agg_preds = shaped_preds.sum(-1) / shaped_preds.size(1) > 0.5
print(f'Test Acc (sample) : {(agg_preds == true_preds).sum() / len(true_preds)}')





