# Please follow the instruction to generate the sentence

## Please follow the below step to generate pseudo sentence

#### Step 1:
Download the prepared region descriptions here. We follow the idea of Xmodal-ctx to generate the region descriptions. If you want to generate region descriptions by yourself, please follow up [CTX](https://github.com/GT-RIPL/Xmodal-Ctx/tree/main/ctx).
#### Step 2:
Use CLIP, BART-Xsum, and Sentence-Bert to generate four pseudo sentences (RaPSG module stage 1)  
```
python generate/clip2sentence.py
```
#### Step 3:
Download the prediction sentences from the frozen captioner here. If you want to get the prediction sentences by yourself, you can try to use four generated pseudo sentences in Step 2 to train the DIFNet and get the prediction sentences.

#### Step 4:
Use LLaMA-7B and the prediction sentences to generate the fifth pseudo sentence.
```
python generate/lora_generate.py
```
After these four steps, we get the five pseudo sentences.
