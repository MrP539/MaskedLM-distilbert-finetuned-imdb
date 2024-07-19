import transformers
import datasets
import numpy as np
import os
import torch
import math
model_checkpoint =os.path.join(r"D:\machine_learning_AI_Builders\บท4\NLP\Mask_filling\MaskedLM-finetuned-imdb\checkpoint-7500")
model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_checkpoint)
  #DistilBERT-base-uncased: ถูกออกแบบมาให้มีขนาดเล็กกว่า  bert-base-uncase โดยมีเลเยอร์ทั้งหมด 6 เลเยอร์, 
  #ขนาดของ hidden units คือ 768, มี attention heads ทั้งหมด 12 heads และมีพารามิเตอร์ประมาณ 66 ล้านพารามิเตอร์ ซึ่งเท่ากับประมาณครึ่งหนึ่งของ BERT-base

tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)

def encoder_funcion(sent):
    result = tokenizer(sent,truncation=True) # truncation=True เป็นการกำหนดให้ tokenizer ทำการตัดข้อความที่มีความยาวเกินค่าที่กำหนดใน model_max_length ให้พอดีกับขีดจำกัดนี้
    return(result)

text = "This is a great [MASK]."

inputs = tokenizer(text=text,return_tensors="pt")
token_logits = model(**inputs).logits
print(inputs["input_ids"])
print(tokenizer.mask_token)

mask_token_id = tokenizer.mask_token_id
print(mask_token_id)
mask_token_index = torch.where(inputs["input_ids"] == mask_token_id)[1]
mask_token_logits = token_logits[0,mask_token_index, : ]
print(mask_token_logits.shape)
 # mask_token_logits คือค่าที่ ทำนาย next word ที่เลือกจาก  mask_token_id

token_top_5 = torch.topk(input=mask_token_logits,k=5,dim=1).indices[0].tolist()   #  indices[num_batch] หมายถึงการดึงข้อมูลจาก batch แรกใน tensor ผลลัพธ์จาก torch.topk
                                                              # ผลลัพธ์จาก torch.topk เป็น tuple ที่มี 2 ค่า:
                                                                # ค่าที่สูงที่สุด values              values=tensor([[7.0727, 6.6514, 6.6425, 6.2530, 5.8618]], grad_fn=<TopkBackward0>), 
                                                                # token_id ของค่าที่สูงที่สุด indices    indices=tensor([[3066, 3112, 6172, 2801, 8658]]))
for i in token_top_5:
    print(f"--> {text.replace(tokenizer.mask_token,tokenizer.decode([i]))}")

 # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#print(tokenizer.decode([1012]))
# 30522 คือการทำนาย next word ของแต่ละคอ 

# def encoder_funcion(sent):
#     result = tokenizer(sent["text"],truncation=True) # truncation=True เป็นการกำหนดให้ tokenizer ทำการตัดข้อความที่มีความยาวเกินค่าที่กำหนดใน model_max_length ให้พอดีกับขีดจำกัดนี้
#     return(result)

# imdb_dataset = datasets.load_dataset("imdb")  # dataset รีวิวหนัง

# tokenized_dataset = imdb_dataset.map(encoder_funcion,batched=True)
# tokenized_dataset = tokenized_dataset.remove_columns(["text","label"])

# tokenized_sample = tokenized_dataset["train"][:3]

# concatenated_samples = {
#     keys:sum(tokenized_sample[keys],[]) for keys in tokenized_sample.keys()
#         # sum() ใน Python สามารถใช้เพื่อรวม lists หลายๆ list เข้าด้วยกัน. 
#         # sum(tokenized_sample[keys], []) จะรวม lists tokenized_sample[keys] เข้าด้วยกันเป็น list เดียว โดยเริ่มจาก list ว่าง [].
# }
# print(f"concatenated_samples: {concatenated_samples}")
# print(f"concatenated_samples_keys: {concatenated_samples.keys()}")
# total_lenght = len(concatenated_samples[list(tokenized_sample.keys())[0]])
# print(f"tokenized_sample: {tokenized_sample.keys()}")
# print(f"tokenized_sample_list: {list(tokenized_sample.keys())}")
# print(f"tokenized_sample_list[0]: {list(tokenized_sample.keys())[0]}")
# print("\n...complete...\n")



# samples = [lm_datasets["train"][i] for i in range(2)]
# batch = whole_word_masking_data_collator(samples)

# for chunk in batch["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'") # หมดปัญหา



# samples = [lm_datasets["train"][i] for i in range(2)]
# for sample in samples:
#     _ = sample.pop("word_ids")
# batch = data_collator(samples)["input_ids"]
# for chunk in batch:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")