import os
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import warnings
import time

warnings.filterwarnings("ignore")

# Set parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 4
max_seq_l = 512
lr = 5e-5
num_epochs = 1
use_cuda = True
model_name = "codet5"
pretrainedmodel_path = "E:/models/codet5-base"  # Path of the pre-trained model
early_stop_threshold = 10
si_lambda = 0.1  # SI正则化项的权重

# Define classes
classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# Define function to read examples
def read_prompt_examples(filename):
    examples = []
    data = pd.read_excel(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    severity = data['severity'].tolist()
    for idx in range(len(data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=int(severity[idx]),
            )
        )
    return examples


# Load model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)

# Define template
template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description: {"placeholder":"text_b"} {"soft":"Classify the severity:"} {"mask"}'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)


# Define function to test the model
def test(prompt_model, test_dataloader, name):
    num_test_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_test_steps))
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            progress_bar.update(1)
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = accuracy_score(alllabels, allpreds)
        precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
        precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
        mcc = matthews_corrcoef(alllabels, allpreds)
        with open(os.path.join('./results', "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                open(os.path.join('./results', "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:
            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')
        print("acc: {}   precisionma: {}  recallma: {} recallwei: {} weighted-f1: {}  macro-f1: {} mcc: {}".format(acc,
                                                                                                                   precisionma,
                                                                                                                   recallma,
                                                                                                                   recallwei,
                                                                                                                   f1wei,
                                                                                                                   f1ma,
                                                                                                                   mcc))
    return acc, precisionma, recallma, f1wei, f1ma


# Define the verbalizer
myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={
                                    "LOW": ["low", "slight"],
                                    "MEDIUM": ["medium", "moderate"],
                                    "HIGH": ["high", "severe"],
                                    "CRITICAL": ["critical", "significant"]
                                })

# Define the prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()
# Define the optimizer and scheduler
# Initialize SI variables
si_omega = {n: torch.zeros_like(p, device='cuda') for n, p in prompt_model.named_parameters() if p.requires_grad}
si_prev_params = {n: p.clone().detach() for n, p in prompt_model.named_parameters() if p.requires_grad}
si_importance = {n: torch.zeros_like(p, device='cuda') for n, p in prompt_model.named_parameters() if p.requires_grad}


loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.01}
]
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)

# 加载五个任务的测试集
test_dataloader1 = PromptDataLoader(
    dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data1/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader2 = PromptDataLoader(
    dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data2/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader3 = PromptDataLoader(
    dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data3/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader4 = PromptDataLoader(
    dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data4/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)
test_dataloader5 = PromptDataLoader(
    dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data5/test.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
    batch_size=batch_size, shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
    decoder_max_length=3)

# Main loop for each dataset
for i in range(1, 3):
    start_time = time.time()
    print("-----------------------第" + str(i) + "次任务---------------------------")
    train_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data" + str(i) + "/train.xlsx"),
        template=mytemplate,
        tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batch_size, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="head",
        decoder_max_length=3)
    validation_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(r"E:/xjc/SOTitlePlus/SOTitlePlus/data" + str(i) + "/valid.xlsx"),
        template=mytemplate,
        tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batch_size, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="head",
        decoder_max_length=3)

    num_training_steps = num_epochs * len(train_dataloader)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)

    if i >= 2:
        prompt_model.load_state_dict(
            torch.load(os.path.join("E:\\xjc\\SOTitlePlus\\SOTitlePlus\\model_code\\result1\\best.ckpt"),
                       map_location=torch.device('cuda:0')))
    # Train and validate the model
    output_dir = "result1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    progress_bar = tqdm(range(num_training_steps))
    bestmetric = 0
    bestepoch = 0
    early_stop_count = 0

    for epoch in range(num_epochs):
        # 训练
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()

            logits = prompt_model(inputs)
            labels = inputs['tgt_text'].cuda()
            loss = loss_func(logits, labels)
            # Train loop
            for epoch in range(num_epochs):
                # train
                tot_loss = 0
                for step, inputs in enumerate(train_dataloader):
                    if use_cuda:
                        inputs = inputs.cuda()

                    logits = prompt_model(inputs)

                    labels = inputs['tgt_text'].cuda()

                    loss = loss_func(logits, labels)

                    # SI regularization term
                    si_loss = 0
                    for n, p in prompt_model.named_parameters():
                        if p.requires_grad:
                            if p.grad is not None:  # Ensure p.grad is not None before updating SI parameters
                                si_importance[n] += (p.grad * (p - si_prev_params[n])).abs().detach()
                                si_prev_params[n] = p.clone().detach()

                    loss += si_lambda * si_loss

                    loss.backward(retain_graph=True)  # Specify retain_graph=True

                    tot_loss += loss.item()
                    optimizer1.step()
                    optimizer1.zero_grad()
                    scheduler1.step()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    scheduler2.step()

                    # Update SI parameters
                    for n, p in prompt_model.named_parameters():
                        if p.requires_grad:
                            if p.grad is not None:  # Ensure p.grad is not None before updating SI parameters
                                si_importance[n] += (p.grad * (p - si_prev_params[n])).abs().detach()
                                si_prev_params[n] = p.clone().detach()

                    progress_bar.update(1)
                print("\nEpoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

        this_epoch_best = False

        end_time = time.time()
        print("用时：", end_time - start_time)

        # 验证
        print('\n\nepoch{}------------validate------------'.format(epoch))
        acc, precision, recall, f1wei, f1mi = test(prompt_model, validation_dataloader, name="dev")
        if f1mi > bestmetric:
            bestmetric = f1mi
            bestepoch = epoch
            this_epoch_best = True
            torch.save(prompt_model.state_dict(), f"{output_dir}/best.ckpt")
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_threshold:
                print("early stopping!!!")
                break



    # Load the best model and test it
    print("----------------------Load the best model and test it-----------------------------")
    prompt_model.load_state_dict(
        torch.load(os.path.join("E:\\xjc\\SOTitlePlus\\SOTitlePlus\\model_code\\result1\\best.ckpt"),
                   map_location=torch.device('cuda:0')))
    print("-------------第" + str(i) + "次任务在第1个数据集上的测试----------------")
    test(prompt_model, test_dataloader1, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第2个数据集上的测试----------------")
    test(prompt_model, test_dataloader2, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第3个数据集上的测试----------------")
    test(prompt_model, test_dataloader3, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第4个数据集上的测试----------------")
    test(prompt_model, test_dataloader4, 'vuldetect_code_summary_test')
    print("-------------第" + str(i) + "次任务在第5个数据集上的测试----------------")
    test(prompt_model, test_dataloader5, 'vuldetect_code_summary_test')


# 只使用SI
