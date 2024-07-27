import torch
import json
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import h5py
import os
import numpy as np
from pycocotools.coco import COCO as pyCOCO
from tqdm import tqdm
import numpy as np
import json
import evaluation


def compute_score(dataloader):
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (sens, caps_gt) in enumerate(iter(dataloader)):
            gen['%d' % (it)] = [sens, ]
            gts['%d' % (it)] = caps_gt
            pbar.update()
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input_ctxt}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
)


def get_coco():
    coco_dic = {}
    roots = {}
    roots['train'] = {
        'img': os.path.join('dataset/coco/', 'train2014'),
        'cap': os.path.join('dataset/coco/annotations/', 'captions_train2014.json')}
    roots['val'] = {
        'img': os.path.join('dataset/coco/', 'val2014'),
        'cap': os.path.join('dataset/coco/annotations/', 'captions_val2014.json')}
    roots['test'] = {
        'img': os.path.join('dataset/coco/', 'val2014'),
        'cap': os.path.join('dataset/coco/annotations/', 'captions_val2014.json')}
    roots['trainrestval'] = {
        'img': (roots['train']['img'], roots['val']['img']),
        'cap': (roots['train']['cap'], roots['val']['cap'])
    }
    ids_set = {}
    ids_set['train'] = np.load(os.path.join('dataset/coco/annotations/', 'coco_train_ids.npy'))
    ids_set['val'] = np.load(os.path.join('dataset/coco/annotations/', 'coco_dev_ids.npy'))
    ids_set['test'] = np.load(os.path.join('dataset/coco/annotations/', 'coco_test_ids.npy'))
    ids_set['trainrestval'] = (
        ids_set['train'],
        np.load(os.path.join(os.path.join('dataset/coco/annotations/', 'coco_restval_ids.npy'))))
    roots['train'] = roots['trainrestval']
    ids_set['train'] = ids_set['trainrestval']
    for split in ['train', 'val', 'test']:
        total = 0
        if isinstance(roots[split]['cap'], tuple):
            coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
            root = roots[split]['img']
        else:
            coco_dataset = (pyCOCO(roots[split]['cap']),)
            root = (roots[split]['img'],)
        ids = ids_set[split]
        if isinstance(ids, tuple):
            bp = len(ids[0])
            ids = list(ids[0]) + list(ids[1])
        else:
            bp = len(ids)
        for index in range(len(ids)):
            if index < bp:
                coco = coco_dataset[0]
            else:
                coco = coco_dataset[1]
            ann_id = ids[index]
            img_id = coco.anns[ann_id]['image_id']
            filename = coco.loadImgs(img_id)[0]['file_name']
            caption = coco.anns[ann_id]['caption']
            if split == 'train':
                if img_id in coco_dic.keys():
                    annotation = coco_dic[img_id]['annotation']
                    annotation.append(caption)
                    coco_dic[img_id]['annotation'] = annotation
                else:
                    coco_dic[img_id] = {'image_path': os.path.join('dataset/coco/images/', filename),
                                        'image_id': img_id, 'annotation': [caption]}
            elif split == 'val':
                if img_id in coco_dic.keys():
                    annotation = coco_dic[img_id]['annotation']
                    annotation.append(caption)
                    coco_dic[img_id]['annotation'] = annotation
                else:
                    coco_dic[img_id] = {'image_path': os.path.join('dataset/coco/images/', filename),
                                        'image_id': img_id, 'annotation': [caption]}
            elif split == 'test':
                if img_id in coco_dic.keys():
                    annotation = coco_dic[img_id]['annotation']
                    annotation.append(caption)
                    coco_dic[img_id]['annotation'] = annotation
                else:
                    coco_dic[img_id] = {'image_path': os.path.join('dataset/coco/images/', filename),
                                        'image_id': img_id, 'annotation': [caption]}
    return coco_dic

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

if __name__ == '__main__':
    us_out = h5py.File('dataset/coco/train_scst_prediction_filter.hdf5', 'r')
    clip = h5py.File('dataset/coco/clip_txt_text.hdf5', 'r')
    coco_dict = get_coco()
    all_keys = []
    for key in coco_dict.keys():
        all_keys.append(key)
    keys = all_keys[0:10000]
    with open("coco_lora_0_10000.json", "a") as json_file:
        dict = []
        with tqdm(desc='Generation', unit='it', total=len(keys)) as pbar:
            for index in keys:
                sample = coco_dict[index]
                img_id = sample['image_id']
                annotation = sample['annotation']
                us_sen = us_out["%d" % img_id][()]
                clip_sens = clip["%d" % img_id][:]
                instruction = "Make a short sentence"
                input_ctxt = "Remake a short sentence based on the sentence: "
                input_ctxt = input_ctxt + bytes.decode(us_sen)
                addition = " with additional information: "
                for txt in clip_sens[0:16]:
                    phase = bytes.decode(txt)
                    addition = addition + phase + ", "
                input_ctxt = input_ctxt + addition
                prompt = generate_prompt(instruction, input_ctxt)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split('Response:')
                prediction = response[1]
                prediction = prediction.split('\n')
                if len(prediction) == 1:
                    sentence = prediction[0]
                else:
                    sentence = prediction[1]
                if len(sentence) > 120:
                    sentence = sentence.split(',')
                    sentence = sentence[0]
                data = coco_dict[img_id]
                data['us_out'] = bytes.decode(us_sen)
                data['lora'] = sentence
                dict.append(data)
                pbar.update()
        json.dump(dict, json_file)
    keys = all_keys[10000:20000]
    with open("coco_lora_10000_20000.json", "a") as json_file:
        dict = []
        with tqdm(desc='Generation', unit='it', total=len(keys)) as pbar:
            for index in keys:
                sample = coco_dict[index]
                img_id = sample['image_id']
                annotation = sample['annotation']
                us_sen = us_out["%d" % img_id][()]
                clip_sens = clip["%d" % img_id][:]
                instruction = "Make a short sentence"
                input_ctxt = "Remake a short sentence based on the sentence: "
                input_ctxt = input_ctxt + bytes.decode(us_sen)
                addition = " with additional information: "
                for txt in clip_sens[0:16]:
                    phase = bytes.decode(txt)
                    addition = addition + phase + ", "
                input_ctxt = input_ctxt + addition
                prompt = generate_prompt(instruction, input_ctxt)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split('Response:')
                prediction = response[1]
                prediction = prediction.split('\n')
                if len(prediction) == 1:
                    sentence = prediction[0]
                else:
                    sentence = prediction[1]
                if len(sentence) > 120:
                    sentence = sentence.split(',')
                    sentence = sentence[0]
                data = coco_dict[img_id]
                data['us_out'] = bytes.decode(us_sen)
                data['lora'] = sentence
                dict.append(data)
                pbar.update()
        json.dump(dict, json_file)
    keys = all_keys[20000:30000]
    with open("coco_lora_20000_30000.json", "a") as json_file:
        dict = []
        with tqdm(desc='Generation', unit='it', total=len(keys)) as pbar:
            for index in keys:
                sample = coco_dict[index]
                img_id = sample['image_id']
                annotation = sample['annotation']
                us_sen = us_out["%d" % img_id][()]
                clip_sens = clip["%d" % img_id][:]
                instruction = "Make a short sentence"
                input_ctxt = "Remake a short sentence based on the sentence: "
                input_ctxt = input_ctxt + bytes.decode(us_sen)
                addition = " with additional information: "
                for txt in clip_sens[0:16]:
                    phase = bytes.decode(txt)
                    addition = addition + phase + ", "
                input_ctxt = input_ctxt + addition
                prompt = generate_prompt(instruction, input_ctxt)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split('Response:')
                prediction = response[1]
                prediction = prediction.split('\n')
                if len(prediction) == 1:
                    sentence = prediction[0]
                else:
                    sentence = prediction[1]
                if len(sentence) > 120:
                    sentence = sentence.split(',')
                    sentence = sentence[0]
                data = coco_dict[img_id]
                data['us_out'] = bytes.decode(us_sen)
                data['lora'] = sentence
                dict.append(data)
                pbar.update()
        json.dump(dict, json_file)
    keys = all_keys[30000:40000]
    with open("coco_lora_30000_40000.json", "a") as json_file:
        dict = []
        with tqdm(desc='Generation', unit='it', total=len(keys)) as pbar:
            for index in keys:
                sample = coco_dict[index]
                img_id = sample['image_id']
                annotation = sample['annotation']
                us_sen = us_out["%d" % img_id][()]
                clip_sens = clip["%d" % img_id][:]
                instruction = "Make a short sentence"
                input_ctxt = "Remake a short sentence based on the sentence: "
                input_ctxt = input_ctxt + bytes.decode(us_sen)
                addition = " with additional information: "
                for txt in clip_sens[0:16]:
                    phase = bytes.decode(txt)
                    addition = addition + phase + ", "
                input_ctxt = input_ctxt + addition
                prompt = generate_prompt(instruction, input_ctxt)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split('Response:')
                prediction = response[1]
                prediction = prediction.split('\n')
                if len(prediction) == 1:
                    sentence = prediction[0]
                else:
                    sentence = prediction[1]
                if len(sentence) > 120:
                    sentence = sentence.split(',')
                    sentence = sentence[0]
                data = coco_dict[img_id]
                data['us_out'] = bytes.decode(us_sen)
                data['lora'] = sentence
                dict.append(data)
                pbar.update()
        json.dump(dict, json_file)
    keys = all_keys[40000:50000]
    with open("coco_lora_40000_50000.json", "a") as json_file:
        dict = []
        with tqdm(desc='Generation', unit='it', total=len(keys)) as pbar:
            for index in keys:
                sample = coco_dict[index]
                img_id = sample['image_id']
                annotation = sample['annotation']
                us_sen = us_out["%d" % img_id][()]
                clip_sens = clip["%d" % img_id][:]
                instruction = "Make a short sentence"
                input_ctxt = "Remake a short sentence based on the sentence: "
                input_ctxt = input_ctxt + bytes.decode(us_sen)
                addition = " with additional information: "
                for txt in clip_sens[0:16]:
                    phase = bytes.decode(txt)
                    addition = addition + phase + ", "
                input_ctxt = input_ctxt + addition
                prompt = generate_prompt(instruction, input_ctxt)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split('Response:')
                prediction = response[1]
                prediction = prediction.split('\n')
                if len(prediction) == 1:
                    sentence = prediction[0]
                else:
                    sentence = prediction[1]
                if len(sentence) > 120:
                    sentence = sentence.split(',')
                    sentence = sentence[0]
                data = coco_dict[img_id]
                data['us_out'] = bytes.decode(us_sen)
                data['lora'] = sentence
                dict.append(data)
                pbar.update()
        json.dump(dict, json_file)
    keys = all_keys[50000:60000]
    with open("coco_lora_50000_60000.json", "a") as json_file:
        dict = []
        with tqdm(desc='Generation', unit='it', total=len(keys)) as pbar:
            for index in keys:
                sample = coco_dict[index]
                img_id = sample['image_id']
                annotation = sample['annotation']
                us_sen = us_out["%d" % img_id][()]
                clip_sens = clip["%d" % img_id][:]
                instruction = "Make a short sentence"
                input_ctxt = "Remake a short sentence based on the sentence: "
                input_ctxt = input_ctxt + bytes.decode(us_sen)
                addition = " with additional information: "
                for txt in clip_sens[0:16]:
                    phase = bytes.decode(txt)
                    addition = addition + phase + ", "
                input_ctxt = input_ctxt + addition
                prompt = generate_prompt(instruction, input_ctxt)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split('Response:')
                prediction = response[1]
                prediction = prediction.split('\n')
                if len(prediction) == 1:
                    sentence = prediction[0]
                else:
                    sentence = prediction[1]
                if len(sentence) > 120:
                    sentence = sentence.split(',')
                    sentence = sentence[0]
                data = coco_dict[img_id]
                data['us_out'] = bytes.decode(us_sen)
                data['lora'] = sentence
                dict.append(data)
                pbar.update()
        json.dump(dict, json_file)
   









