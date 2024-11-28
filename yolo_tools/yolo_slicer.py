import pathlib
from typing import Dict, List, Tuple

import cv2
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import os
import shutil
import random

from roiyolowd.reassembler import Reassembler
from roiyolowd.util import Rect, split_list_randomly, sample_rects

ranks = {
    "Weed_Crop_thin_star": 0,
    "Weed_small": 1,
    "Barley": 2,
    "Weed_weed": 3,
    "Weed_balls_wide": 4,
    "Weed": 5
}

label_mapping_0 = {
    "Barley": "Barley",
    "Weed_Crop_thin_star": "Weed_Crop_thin_star",
    "Weed_balls_wide": "Weed_balls_wide",
    "Weed_dense": "Weed_dense",
    "Weed_fluffy": "Weed_fluffy",
    "Weed_small": "Weed_small",
    "Weed_thin": "Weed_thin",
    "Weed_weed": "Weed_weed"
}

label_mapping_cropweed = {
    "Barley": "Barley",
    "Weed_Crop_thin_star": "Weed",
    "Weed_balls_wide": "Weed",
    "Weed_dense": "Weed",
    "Weed_fluffy": "Weed",
    "Weed_small": "Weed",
    "Weed_thin": "Weed",
    "Weed_weed": "Weed"
}

label_mapping_crop_weeds1 = {
    "Barley": "Barley",
    "Weed_Crop_thin_star": "Weed_Crop_thin_star",
    "Weed_balls_wide": "Weed_small",
    "Weed_dense": "Weed_small",
    "Weed_fluffy": "Weed_small",
    "Weed_small": "Weed_small",
    "Weed_thin": "Weed_small",
    "Weed_weed": "Weed_weed"
}


class BBox:
    __slots__ = ["cx", "cy", "w", "h"]

    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h


def rect2bbox(r: Rect):
    return BBox(r.x + r.w / 2, r.y + r.h / 2, r.w, r.h)


def bboxnorm(bbox: BBox, imgw: int, imgh: int):
    return BBox(bbox.cx / imgw, bbox.cy / imgh, bbox.w / imgw, bbox.h / imgh)


class LabelMapper:
    """
    Given the original classes.txt and mapping rules, generate new class ids and mappings from old class ids to new class ids.
    """

    def __init__(self, mappings: Dict[str, str], labels: List[str]):
        """
        Args:
            mappings: Dictionary of label mappings, mappings from original name to new name.
            labels: lines from classes.txt
        """
        self.mappings: Dict[str, str] = mappings
        self.labels: List[str] = labels
        self.mappings_oldv2idx: Dict[str, int] = {}
        self.mappings_newv2idx: Dict[str, int] = {}
        self.mappings_i2i: Dict[int, int] = {}
        self.newclasslines = ""
        self.__init()

    def __init(self):
        for i, lbl in enumerate(self.labels):
            self.mappings_oldv2idx[lbl] = i
        values = list(set(self.mappings.values()))
        values.sort(key = lambda x: ranks[x])
        new_labels_list: List[str] = []
        for i, v in enumerate(values):
            self.mappings_newv2idx[v] = i
            new_labels_list.append(v)
        for lbl in self.labels:
            oldidx = self.mappings_oldv2idx[lbl]
            newlbl = self.mappings[lbl]
            newidx = self.mappings_newv2idx[newlbl]
            self.mappings_i2i[oldidx] = newidx
        self.newclasslines = '\n'.join(new_labels_list)

    def map(self, label_cls: int) -> int:
        return self.mappings_i2i[label_cls]

    def printinfo(self):
        for lbl in self.labels:
            newlbl = self.mappings[lbl]
            print(f"[{self.mappings_oldv2idx[lbl]}]{lbl} -> [{self.mappings_newv2idx[newlbl]}]{newlbl}")


def make_gallery(im: np.ndarray, boxes_rect: List[Tuple[Rect, int]], bg_count: int, suffix: str, newpath: str,
                 imgname: str, ext: str):
    # We split the list of bboxes into smaller sublists to avoid large output images.
    splitted_rect_list = split_list_randomly(boxes_rect.copy(), (len(boxes_rect) + 9) // 15)  # list[list[(Rect,cls)]]
    all_rects = [rect for rect, cls in boxes_rect]

    # And we generate compilations consisting solely of backgrounds, which is denoted by a null sublist
    for i in range(bg_count):
        splitted_rect_list.append(None)

    for li, lbl_list in enumerate(splitted_rect_list):
        bg_compliation = lbl_list is None
        if bg_compliation:
            lbl_list = []

        ra = Reassembler()
        for rect, cls in lbl_list:
            ra.addRect(rect)
        rect_list = [rect for rect, cls in lbl_list]
        cls_list = [cls for rect, cls in lbl_list]

        # add background rects
        bg_samples = sample_rects(im.shape[1], im.shape[0], all_rects, scale=20,
                                  num_samples=80 if bg_compliation else len(rect_list) * 3)
        for rect in bg_samples:
            ra.addRect(rect)

        # We use variable margin size so that the models don't make decisions based on margins
        imgr = ra.reassemble(im, use_resizable_packer=True, border_thickness=2, padding_size=(0, 25), roi_extractor=None)
        mapped = ra.mapping(rect_list)
        mapped_bbox = [rect2bbox(rm.dst) for rm in mapped]

        lbl_table = []
        for i in range(len(mapped_bbox)):
            bbox = mapped_bbox[i]
            xywhn = bboxnorm(bbox, imgr.shape[1], imgr.shape[0])
            lbl_table.append([cls_list[i], xywhn.cx, xywhn.cy, xywhn.w, xywhn.h])

        slice_path = f"{newpath}/images/{imgname.replace(ext, f'_r_{li}_{suffix}{ext}')}"
        slice_labels_path = f"{newpath}/labels/{imgname.replace(ext, f'_r_{li}_{suffix}.txt')}"
        Image.fromarray(imgr).save(slice_path)
        lbl_df = pd.DataFrame(lbl_table, columns=['class', 'x1', 'y1', 'w', 'h'])
        lbl_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')


def make_gallery_tile_only(im: np.ndarray, boxes_rect: List[Tuple[Rect, int]], bg_count: int, suffix: str, newpath: str,
                 imgname: str, ext: str):

    for li, lbl_list in enumerate([boxes_rect]):
        lbl_list = lbl_list.copy()
        random.shuffle(lbl_list)
        lbl_list = lbl_list[len(lbl_list) // 4:]

        ra = Reassembler()
        for rect, cls in lbl_list:
            ra.addRect(rect)
        rect_list = [rect for rect, cls in lbl_list]
        cls_list = [cls for rect, cls in lbl_list]

        # We use variable margin size so that the models don't make decisions based on margins
        # imgr = ra.reassemble(im, autosize=True, border=2, margin=(0, 25), roi_extractor=None)  # Randomized margin saves the world
        imgr = ra.reassemble(im, use_resizable_packer=True, border_thickness=2, padding_size=8, roi_extractor=None)
        mapped = ra.mapping(rect_list)
        mapped_bbox = [rect2bbox(rm.dst) for rm in mapped]

        lbl_table = []
        for i in range(len(mapped_bbox)):
            bbox = mapped_bbox[i]
            xywhn = bboxnorm(bbox, imgr.shape[1], imgr.shape[0])
            lbl_table.append([cls_list[i], xywhn.cx, xywhn.cy, xywhn.w, xywhn.h])

        slice_path = f"{newpath}/images/{imgname.replace(ext, f'_r_{li}_{suffix}{ext}')}"
        slice_labels_path = f"{newpath}/labels/{imgname.replace(ext, f'_r_{li}_{suffix}.txt')}"
        Image.fromarray(imgr).save(slice_path)
        lbl_df = pd.DataFrame(lbl_table, columns=['class', 'x1', 'y1', 'w', 'h'])
        lbl_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

def relabel(imnames, labnames, newpath, slice_size, ext, label_mapper: LabelMapper):
    check_dir(os.path.join(newpath, 'images'))
    check_dir(os.path.join(newpath, 'labels'))
    for imname in imnames:
        imname = imname.replace('\\', '/')
        filename = imname.split("/")[-1]
        labname = pathlib.Path(imname)
        labname = labname.parent.with_name("labels").joinpath(labname.name.replace(ext, '.txt'))
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        for index in range(len(labels)):
            cls = int(labels.at[index, 'class'])
            labels.at[index, 'class'] = label_mapper.map(cls)

        labels.to_csv(str(os.path.join(newpath, 'labels', filename.replace(ext, '.txt'))), sep=' ', index=False,
                      header=False)
        shutil.copy(imname, str(os.path.join(newpath, 'images', filename)))


def tiler(imnames, labnames, newpath, slice_size, ext, label_mapper: LabelMapper):
    for imname in imnames:
        imname = imname.replace('\\', '/')
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        if imr.shape[2] == 4:
            imr = cv2.cvtColor(imr, cv2.COLOR_BGRA2BGR)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = pathlib.Path(imname)
        labname = labname.parent.with_name("labels").joinpath(labname.name.replace(ext, '.txt'))
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        filename = imname.split('/')[-1]

        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height

        boxes = []
        boxes_rect: List[Tuple[Rect, int]] = []

        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w'] / 2
            y1 = row[1]['y1'] - row[1]['h'] / 2
            x2 = row[1]['x1'] + row[1]['w'] / 2
            y2 = row[1]['y1'] + row[1]['h'] / 2

            cls = int(row[1]['class'])
            cls = label_mapper.map(cls)

            boxes.append((cls, Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
            boxes_rect.append((Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), cls))

        counter = 0
        print('Image:', imname)
        # create tiles and find intersection with bounding boxes for each tile
        for i in range(((height + slice_size - 1) // slice_size)):
            for j in range(((width + slice_size - 1) // slice_size)):
                x1 = j * slice_size
                y1 = i * slice_size
                x2 = min((j + 1) * slice_size, width)
                y2 = min((i + 1) * slice_size, height)
                piece_width = x2 - x1
                piece_height = y2 - y1

                # if piece_width<slice_size/2 or piece_height<slice_size/2:
                #     continue

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])

                        # get the smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope

                        centre = new_box.centroid
                        x, y = new_box.exterior.coords.xy

                        new_width = (max(x) - min(x)) / piece_width
                        new_height = (max(y) - min(y)) / piece_height

                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / piece_width
                        new_y = (centre.coords.xy[1][0] - y1) / piece_height

                        counter += 1

                        label_cls = box[0]
                        remaining_area = inter.area / box[1].area

                        if remaining_area > 0.3:
                            slice_labels.append([label_cls, new_x, new_y, new_width, new_height])

                sliced = imr[y1:y2, x1:x2]
                sliced_im = Image.fromarray(sliced)

                slice_path = f"{newpath}/images/{filename.replace(ext, f'_{i}_{j}{ext}')}"
                slice_labels_path = f"{newpath}/labels/{filename.replace(ext, f'_{i}_{j}.txt')}"
                print(slice_path)
                os.makedirs(os.path.dirname(slice_path), exist_ok=True)
                os.makedirs(os.path.dirname(slice_labels_path), exist_ok=True)
                sliced_im.save(slice_path)

                slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                print(slice_df)
                slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
        # end of loop

        for i in range(10):
            make_gallery_tile_only(imr, boxes_rect, 10, f"{i}", newpath, filename, ext)


def splitter(dataset_path, output_dir, train_ratio=0.8):
    input_images_path = os.path.join(dataset_path, 'images')
    input_labels_path = os.path.join(dataset_path, 'labels')

    output_images_path = os.path.join(output_dir, 'images')
    output_labels_path = os.path.join(output_dir, 'labels')

    # Create directories
    os.makedirs(os.path.join(output_images_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_images_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_labels_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_labels_path, 'val'), exist_ok=True)

    # Move images and labels to train/val directories
    # Adjust this section based on your dataset split (train/val)
    allfiles = []
    for filename in os.listdir(os.path.join(dataset_path, "images")):
        filename = filename.replace(".JPG", ".jpg").replace(".PNG", ".png")
        if filename.endswith('.jpg') or filename.endswith('.png'):
            allfiles.append(filename)

    random.shuffle(allfiles)
    split_index = int(train_ratio * len(allfiles))
    train_files = allfiles[:split_index]
    val_files = allfiles[split_index:]

    for filename in train_files:
        shutil.copy(os.path.join(input_images_path, filename), os.path.join(output_images_path, 'train', filename))
        label_file = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        shutil.copy(os.path.join(input_labels_path, label_file), os.path.join(output_labels_path, 'train', label_file))

    for filename in val_files:
        shutil.copy(os.path.join(input_images_path, filename), os.path.join(output_images_path, 'val', filename))
        label_file = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        shutil.copy(os.path.join(input_labels_path, label_file), os.path.join(output_labels_path, 'val', label_file))

    with open(os.path.join(dataset_path, 'classes.txt'), 'r') as f:
        lns = f.readlines()
        lns = [cl.replace('\n', '').replace('\r', '') for cl in lns]

    # Create data.yaml file
    data_yaml_content = f"""
path: {pathlib.Path(output_dir).absolute()}
train: images/train
val: images/val

nc: {len(lns)}
names: {lns}
    """

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml_content)


def check_dir(target):
    if not os.path.exists(target):
        os.makedirs(target)
    elif len(os.listdir(target)) > 0:
        raise Exception(f"{target} folder should be empty")


def tile_and_split(source, label_mappings: Dict[str, str], ext=".png", size=600, ratio=0.8, split_only=False):
    imnames = glob.glob(f'{source}/images/*{ext}')
    labnames = glob.glob(f'{source}/labels/*.txt')
    classestxt = f"{source}/classes.txt"

    target = pathlib.Path(source)
    target = target.with_name(target.name + "_splitted")
    target_final = pathlib.Path(source)
    target_final = target_final.with_name(target_final.name + "_final")
    target_relabelled = pathlib.Path(source)
    target_relabelled = target_relabelled.with_name(target_relabelled.name + "_relabelled")
    classeslines = ""
    with open(classestxt, 'r') as f:
        classeslines = f.readlines()
        classeslines = [s.replace('\n', '').replace('\r', '') for s in classeslines]

    label_mapper = LabelMapper(label_mappings, classeslines)
    label_mapper.printinfo()

    print(target)

    if len(imnames) == 0:
        raise Exception("Source folder should contain some images")
    elif len(imnames) != len(labnames):
        raise Exception("Dataset should contain equal number of images and txt files with labels")

    if not split_only:
        check_dir(target)
        check_dir(target_relabelled)

    if not split_only:
        with open(target_relabelled.joinpath('classes.txt'), 'w') as f:
            f.write(label_mapper.newclasslines)
        relabel(imnames, labnames, target_relabelled, size, ext, label_mapper)

        with open(target.joinpath('classes.txt'), 'w') as f:
            f.write(label_mapper.newclasslines)
        tiler(imnames, labnames, target, size, ext, label_mapper)

    # classes.names should be located one level higher than images
    # this file is not changing, so we will just copy it to a target folder
    check_dir(target_final)
    splitter(target, target_final, ratio)

def convert_to_trainable(source, ratio=0.8):
    target_final = pathlib.Path(source)
    target_final = target_final.with_name(target_final.name + "_final")
    check_dir(target_final)
    splitter(source, target_final, ratio)



if __name__ == "__main__":
    #convert_to_trainable("../dataset/test3_relabelled", 0.7)
    tile_and_split("../dataset/test3_2", label_mappings=label_mapping_crop_weeds1, split_only=False)
