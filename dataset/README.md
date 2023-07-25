# Please put all data files here

## The file structure need be structed as follows:
'''
dataset
    |
    +-- annotations (folder)
    |       |
    |       +-- "captions_train2014.json"
    |       +-- "captions_val2014.json"
    |       +-- "coco_dev_ids.npy"
    |       +-- ....
    |       |
    +-- segmentations (folder)
    |       |
    |       +-- train2014 (folder)
    |       +-- val2014 (folder)
    |       |
    +-- coco_all_align.hdf5
    |       |
    +-- clip_feat.hdf5
    |       |
    +-- pseudo_sentence.hdf5
    |       |

'''