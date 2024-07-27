#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:35:03 2022

@author: zhiyuan
"""
from sentence_transformers import SentenceTransformer, util
import h5py
import torch
from transformers import pipeline
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=0)
device = torch.device('cuda')
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1').to(device)
#clip_text_path = "datasets/CLIP/clip_txt_text.hdf5"
clip_text_path = "datasets/CLIP/clip_txt_text.hdf5"
clip = h5py.File(clip_text_path, 'r')

if __name__ == '__main__':
    indexs = clip.keys()
    with h5py.File("pseudo_sentences.hdf5", 'w') as w:
        for img_id in indexs:
            img_id = int(img_id)
            clip_text = clip['%d' % img_id][:]
            clip_text1 = ""
            clip_text2 = ""
            clip_text3 = ""
            clip_text4 = ""
            #generate pseudo sentence 1
            for i in clip_text[0:4]:
                clip_text1 = clip_text1 +" "+bytes.decode(i)
            clip_text1 = clip_text1.split()
            clip_text1 = (" ".join(sorted(set(clip_text1), key=clip_text1.index)))
            summary1 = summarizer(clip_text1)
            ps1 = summary1[0]['summary_text']
            # generate the rest 3 pseudo sentences
            docs = []
            for txt in clip_text[4:]:
                docs.append(bytes.decode(txt))
            query_emb = model.encode(ps1)
            doc_emb = model.encode(docs)
            scores = util.dot_score(query_emb, doc_emb)[0].cuda().tolist()
            doc_score_pairs = list(zip(docs, scores))
            doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            for i in range(4):
                clip_text2 = clip_text2 + " " + doc_score_pairs[i][0]
                clip_text3 = clip_text3 + " " + doc_score_pairs[i+4][0]
                clip_text4 = clip_text4 + " " + doc_score_pairs[i+8][0]
            clip_text2 = clip_text2.split()
            clip_text2 = (" ".join(sorted(set(clip_text2), key=clip_text2.index)))
            clip_text3 = clip_text3.split()
            clip_text3 = (" ".join(sorted(set(clip_text3), key=clip_text3.index)))
            clip_text4 = clip_text4.split()
            clip_text4 = (" ".join(sorted(set(clip_text4), key=clip_text4.index)))
            summary2 = summarizer(clip_text2)
            ps2 = summary2[0]['summary_text']
            summary3 = summarizer(clip_text3)
            ps3 = summary3[0]['summary_text']
            summary4 = summarizer(clip_text4)
            ps4 = summary4[0]['summary_text']
            sentences = [ps1,ps2,ps3,ps4]
            print(sentences)
            w[str(img_id)] = sentences







