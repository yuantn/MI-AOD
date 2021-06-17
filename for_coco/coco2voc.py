from pycocotools.coco import COCO
from pascal_voc_writer import Writer
import argparse
import os

def coco2voc(ann_file, output_dir):
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']
    txtfile = open(output_dir[:-12] + 'ImageSets/Main/trainval.txt', mode='w+')
    for img in coco.imgs:
        catIds = coco.getCatIds()
        annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)
        if len(annIds) > 0:
            img_fname = coco.imgs[img]['file_name']
            image_fname_ls = img_fname.split('.')
            txtfile.write(image_fname_ls[0] + '\n')
            image_fname_ls[-1] = 'xml'
            label_fname = '.'.join(image_fname_ls)
            writer = Writer(img_fname, coco.imgs[img]['width'], coco.imgs[img]['height'])
            anns = coco.loadAnns(annIds)
            for a in anns:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]
                catname = cat_idx[a['category_id']]
                writer.addObject(catname, bbox[0], bbox[1], bbox[2], bbox[3])
                writer.save(output_dir+'/'+label_fname)
    txtfile.close()


parser = argparse.ArgumentParser(description='Convert COCO annotations to PASCAL VOC XML annotations')
parser.add_argument('--ann_file',
                    help='Path to annotations file')
parser.add_argument('--output_dir',
                    help='Path to output directory where annotations are to be stored')
args = parser.parse_args()
try:
    os.mkdir(args.output_dir)
except FileExistsError:
    pass

coco2voc(ann_file=args.ann_file, output_dir=args.output_dir)