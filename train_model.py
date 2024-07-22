# ทำไมเราต้องเทรนการเติมคำ (Mask filling)
#ในปี 2018, มีงานวิจัยงานหนึ่งที่ชื่อว่า ULMfit ได้มีแนวคิดเกี่ยวกับ language model ในการเพิ่มประสิทธิภาพโดยใช้เทคนิค "การย้ายโดเมน" (domain adaptation) 
# โดยที่ขั้นแรกจะต้องให้โมเดลเทรนบนโดเมนทั่วๆไป (เช่น Wikipedia) จากนั้นก็เทรนแบบเดิมอีกครั้ง แต่ทำบนดาต้าเซ็ตที่สนใจ (เช่น รีวิวหนัง) 
# จากในนั้นขั้นตอนสุดท้ายก็ Finetune บนงานที่สนใจ ทำให้ language model มีประสิทธิภาพสูงสุด!!

import transformers
import datasets
import numpy as np
import os
import collections
import pandas as pd
import math
import torch


###########################################################################   setup dataset  #####################################################################################

model_checkpoint = "distilbert-base-uncased"

tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)

def encoder_funcion(sent):
    result = tokenizer(sent["text"],truncation=True) # truncation=True เป็นการกำหนดให้ tokenizer ทำการตัดข้อความที่มีความยาวเกินค่าที่กำหนดใน model_max_length ให้พอดีกับขีดจำกัดนี้
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return(result)

imdb_dataset = datasets.load_dataset("imdb")  # dataset รีวิวหนัง

tokenized_dataset = imdb_dataset.map(encoder_funcion,batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text","label"])

#**********************  การแบ่่ง data เป็น chunk  **************************

#  กำหนด maximum_lenght (มากสุดกี่คำต่อ 1 inputs) **แต่**
# maximum_lenght ที่มากจะต้องการการประมวลผลที่กินพลังงานมาก โดยจากค่า defalse  distilbert-base-uncased  maximum_lenght = 512 /inputs  ถ้า inputs เกิน maximum_lenght จะเกิด error
# แต่โดยปกติ dataset imdb batch=2 // 1 batch ก็เกิน 512 แล้ว จึงต้องแบ่งเป็น chunk(ก้อน) ก่อน

chunk_size = 32

def group_chunk_size(inputs_data):

    #รวมข้อมูลใน inputs
    concatenated_data = {keys:sum(inputs_data[keys],[]) for keys in inputs_data.keys()}
        # sum() ใน Python สามารถใช้เพื่อรวม lists หลายๆ list เข้าด้วยกัน. 
        # sum(tokenized_sample[keys], []) จะรวม lists tokenized_sample[keys] เข้าด้วยกันเป็น list เดียว โดยเริ่มจาก list ว่าง [].

    #วัดความยาวใน input ทั้งหมด
    total_lenght_ = len(concatenated_data["input_ids"])
        
    # เราจะเลือกใช้ drop เศษทิ้ง หมายถึงเลือก chunk 1 จน ถึงอันก่อนสุด (ถ้า chunk สุดท้ายเล็กกว่า chunk_size  ที่กำหนดไว้)
    total_lenght = (total_lenght_//chunk_size)*chunk_size 
        # ปกติการแบ่งเป็นก้อนๆมักจะมีเศษอยู่ที่ก้อนสุดท้าย (จากตัวอย่างจะเหลือแค่ 32 คำ จาก 128 คำ) โดยเราจะเก็บไว้ก็ได้ (แต่ต้องไป padding ซึ่งออกจะยุ่งยาก) 
        # เพียงแค่ในบางตัวอย่างมันอาจจะเหลือแค่คำสองคำ ทำให้ไม่เหลือข้อมูลอะไรให้โมเดลได้เรียนรู้มากนัก ทำให้คนจึงนิยมที่จะตัดออก
    result = {keys:[values[i:i+chunk_size] for i in range(0,total_lenght,chunk_size)] for keys,values in concatenated_data.items()}
    # Create labels
    result["labels"] = result["input_ids"].copy()
    return (result)


lm_dataset = tokenized_dataset.map(group_chunk_size,batched=True)

print(lm_dataset)


#******************** แทนที่คำด้วย [MASK] ให้โมเดลทำนาย *************************

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) # ใน 1 ประโยค จะให้ทำนายคำแค่ 15% ของทั้งหมด
# *** แต่ ***
# แต่ปัญหาตอนนี้คือตอนที่เราแทนคำด้วย [MASK] มันดันแทนแค่คำย่อยของคำนั้นๆ (sub-word) ไม่ได้แทนทั้งหมดเช่น "bromwell => [MASK]mwell"

#------- ฉนั้นเราจะต้องทำในสิ่งที่เรียกว่า "whole-word-mask" --------

wwm_probability = 0.2 # %การ MASK

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return transformers.default_data_collator(features)

#**************  down size data  *******************
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) # ใน 1 ประโยค จะให้ทำนายคำแค่ 15% ของทั้งหมด
train_set_size = 10000
val_set_size = int(0.1*train_set_size)

down_dataset = lm_dataset["train"].train_test_split(test_size=val_set_size,train_size=train_set_size,seed=42)

print(down_dataset)
print(len(down_dataset["train"]))


#####################################################################################  Create model  ###############################################################################################

model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_checkpoint)
  #DistilBERT-base-uncased: ถูกออกแบบมาให้มีขนาดเล็กกว่า  bert-base-uncase โดยมีเลเยอร์ทั้งหมด 6 เลเยอร์, 
  #ขนาดของ hidden units คือ 768, มี attention heads ทั้งหมด 12 heads และมีพารามิเตอร์ประมาณ 66 ล้านพารามิเตอร์ ซึ่งเท่ากับประมาณครึ่งหนึ่งของ BERT-base

    #ในตอนที่โมเดลได้รับ input_ids จะมีการทำ Embedding ภายในโมเดลโดยอัตโนมัติเมื่อเราทำการฝึกหรือประเมินผล:


#####################################################################################  Train  ###############################################################################################

output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class CSVLogger(transformers.TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
        self.log_history.append(logs)
        self.save_log()

    def save_log(self):
        df = pd.DataFrame(self.log_history)
        df.to_csv(os.path.join(self.output_dir, "training_log.csv"), index=False)

def compute_metrics(eval_pred):
    print("Eval_pred:",eval_pred)
    logits, labels = eval_pred
    #predictions = np.argmax(logits, axis=-1)

    print("Logits type:", type(logits), "Logits shape:", logits.shape)
    print("Labels type:", type(labels), "Labels shape:", labels.shape)

    logits = torch.tensor(logits)  # แปลงเป็นเทนเซอร์ถ้าจำเป็น
    labels = torch.tensor(labels)  # แปลงเป็นเทนเซอร์ถ้าจำเป็น
    logits = logits.view(-1, 30522)  # ปรับขนาด logits
    labels = labels.view(-1)  # ปรับขนาด labels
    eval_loss = torch.nn.functional.cross_entropy(logits, labels)
    #eval_loss = float(torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).mean())
    perplexity = math.exp(eval_loss) if eval_loss is not None else float('inf')
    test = "test_Eval_metrict"
    return {"eval_loss": eval_loss, "perplexity": perplexity}


csv_logger = CSVLogger(output_dir="results")

batch_size = 4
#************* Create Training Argument *************
logging_step = len(down_dataset["train"])//batch_size
#print(logging_step)
print("\n...Training...\n")

training_argument = transformers.TrainingArguments(
    output_dir="MaskedLM-finetuned-imdb",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    fp16=True,
    learning_rate=5e-5,
    logging_steps=logging_step,
    load_best_model_at_end=True,
    push_to_hub=True

)

trainer = transformers.Trainer(
    model,
    args=training_argument,
    train_dataset=down_dataset["train"],
    eval_dataset=down_dataset["test"],
    data_collator=whole_word_masking_data_collator,
    callbacks=[csv_logger],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    
)

# eval_results = trainer.evaluate()
# print(f"\n>>> Befor Perplexity: {math.exp(eval_results['eval_loss']):.2f}\n") # โมเดลก่อนเทรน

trainer.train()

eval_results = trainer.evaluate()
print(f"\n>>> Afer Perplexity: {math.exp(eval_results['eval_loss']):.2f}\n") # โมเดลหลังเทรน


