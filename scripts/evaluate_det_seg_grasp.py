import argparse
import os
import numpy as np
import scipy
import cv2
from functools import partial
from shapely.geometry import Polygon
import torch
import torch.optim as optim
import torch.utils.data as data
from torch import distributed

import grasp_det_seg.models as models
from grasp_det_seg.algos.detection import PredictionGenerator, ProposalMatcher, DetectionLoss
from grasp_det_seg.algos.fpn import DetectionAlgoFPN, RPNAlgoFPN
from grasp_det_seg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from grasp_det_seg.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from grasp_det_seg.config import load_config
from grasp_det_seg.data_OCID import iss_collate_fn, OCIDTestDataset, OCIDTestTransform, read_boxes_from_file,\
    prepare_frcnn_format, CornellDataset
from grasp_det_seg.data_OCID.OCID_class_dict import colors_list, cls_list
from grasp_det_seg.data_OCID.sampler import DistributedARBatchSampler
from grasp_det_seg.models.det_seg import DetSegNet_OCID, DetSegNet_Cornell
from grasp_det_seg.modules.fpn import FPN, FPNBody
from grasp_det_seg.modules.fusion import FusionModule
from grasp_det_seg.modules.heads import RPNHead, FPNROIHead, FPNSemanticHeadDeeplab
from grasp_det_seg.utils import logging
from grasp_det_seg.utils.meters import AverageMeter
from grasp_det_seg.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS
from grasp_det_seg.utils.parallel import DistributedDataParallel
from grasp_det_seg.utils.snapshot import resume_from_snapshot
from skimage.measure import regionprops

parser = argparse.ArgumentParser(description="OCID detection and segmentation test script")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--log_dir", type=str, default=".", help="Write logs to the given directory")
parser.add_argument("config", metavar="FILE", type=str, help="Path to configuration file")
parser.add_argument("model", metavar="FILE", type=str, help="Path to model file")
parser.add_argument("data", metavar="DIR", type=str, help="Path to dataset")
parser.add_argument("out_dir", metavar="DIR", type=str, help="Path to output directory")
parser.add_argument("dataset_name", type=str, help="Which dataset will be used")


def save_param_file(writer, param_file):
    data_sum = ''
    with open(param_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            data_sum += line + '  \n'
    writer.add_text('dataset_parameters', data_sum)
    return


def ensure_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass


def Rotate2D(pts, cnt, ang):
    ang = np.deg2rad(ang)
    return scipy.dot(pts - cnt, scipy.array([[scipy.cos(ang), scipy.sin(ang)], [-scipy.sin(ang),
                                                                                scipy.cos(ang)]])) + cnt


def calc_jacc_IOU(gt_boxes, gt_theta_, r_bbox__best, theta_best):
    ret_val = 0
    for cnt in range(gt_boxes.shape[0]):
        try:
            gt_pts = gt_boxes[cnt].T
            gt_theta = gt_theta_[cnt]
            pts_rot = np.asarray(r_bbox__best)

            # check angle difference
            if np.abs(gt_theta - theta_best) < 30 or \
                    (np.abs(np.abs(gt_theta - theta_best) - 180.)) < 30:
                pts_rot = Polygon(pts_rot)
                gt_pts = Polygon(gt_pts)

                intersect = gt_pts.intersection(pts_rot).area / gt_pts.union(pts_rot).area
                print('real'+str(intersect))
                if intersect > .25:
                    ret_val = 1
                    break
        except:
            continue
    return ret_val


def save_prediction_image_Cornell(img_raw, raw_pred, img_abs_path, img_root_path, im_size, out_dir):
    num_classes_theta = 18
    threshold = 0
    total_boxes = 0
    correct_boxes = 0

    for i, (bbx_pred, cls_pred, obj_pred, iou_pred) in enumerate(zip(
            raw_pred["bbx_pred"], raw_pred["cls_pred"], raw_pred["obj_pred"], raw_pred["iou_pred"])):

        img_path = os.path.join(img_root_path[i], img_abs_path[i])
        im_size_ = im_size[i]
        ensure_dir(out_dir)
        _, im_name = os.path.split(img_abs_path[i])
        out_path = os.path.join(out_dir, im_name[:-4] + ".png")
        img_origin = cv2.imread(img_path)
        img = img_raw[i]
        img_best_boxes = np.copy(img)
        img_all_boxes = np.copy(img)
        img_gt_boxes = np.copy(img)
        delta_xy = np.array([[int(img_origin.shape[1] / 2 - int(im_size_[1] / 2))],
                             [int(img_origin.shape[0] / 2 - int(im_size_[0] / 2))]])

        # img_cls = np.copy(img)

        gt_boxes_path = img_path.replace('r.png', 'cpos.txt')
        gt_boxes = read_boxes_from_file(gt_boxes_path, delta_xy)

        print('gt_boxes = ' + str(len(gt_boxes)))
        if len(gt_boxes) == 0:
            print('txt file empty')
            continue
        else:
            total_boxes += 1

            if bbx_pred is None:
                continue
            else:
                (gt_boxes_, gt_theta, cls) = prepare_frcnn_format(gt_boxes, im_size_)

                for box in gt_boxes_:
                    pt1 = (int(box[0][0]), int(box[1][0]))
                    pt2 = (int(box[0][1]), int(box[1][1]))
                    pt3 = (int(box[0][2]), int(box[1][2]))
                    pt4 = (int(box[0][3]), int(box[1][3]))
                    
                    cv2.line(img_gt_boxes, pt1, pt2, (255, 0, 0), 2)
                    cv2.line(img_gt_boxes, pt2, pt3, (0, 0, 255), 2)
                    cv2.line(img_gt_boxes, pt3, pt4, (255, 0, 0), 2)
                    cv2.line(img_gt_boxes, pt4, pt1, (0, 0, 255), 2)
                
                best_confidence = 0.
                r_bbox__best = None
                theta_best = None
                cnt_best = None
                for bbx_pred_i, cls_pred_i, obj_pred_i, iou_pred_i in zip(bbx_pred, cls_pred, obj_pred, iou_pred):
                    obj_iou = 0.3*(iou_pred_i.item()) + 0.7*(obj_pred_i.item())
                    if obj_iou > threshold:
                        pt1 = (int(bbx_pred_i[0]), int(bbx_pred_i[1]))
                        pt2 = (int(bbx_pred_i[2]), int(bbx_pred_i[3]))
                        
                        cls = cls_pred_i.item()
                        theta = ((180 / num_classes_theta) * cls) + 5
                        
                        pts = scipy.array([[pt1[0], pt1[1]], [pt2[0], pt1[1]], [pt2[0], pt2[1]], [pt1[0], pt2[1]]])
                        cnt = scipy.array([(int(bbx_pred_i[0]) + int(bbx_pred_i[2])) / 2,
                                            (int(bbx_pred_i[1]) + int(bbx_pred_i[3])) / 2])
                        r_bbox_ = Rotate2D(pts, cnt, 90 - theta)
                        r_bbox_ = r_bbox_.astype('int16')

                        if (int(cnt[1]) >= im_size_[0]) or (int(cnt[0]) >= im_size_[1]):
                            continue

                        if obj_iou >= best_confidence:
                            best_confidence = obj_iou
                            r_bbox__best = r_bbox_
                            theta_best = theta
                            cnt_best = cnt

                        cv2.line(img_all_boxes, tuple(r_bbox_[0]), tuple(r_bbox_[1]), (255, 0, 0), 2)
                        cv2.line(img_all_boxes, tuple(r_bbox_[1]), tuple(r_bbox_[2]), (0, 0, 255), 2)
                        cv2.line(img_all_boxes, tuple(r_bbox_[2]), tuple(r_bbox_[3]), (255, 0, 0), 2)
                        cv2.line(img_all_boxes, tuple(r_bbox_[3]), tuple(r_bbox_[0]), (0, 0, 255), 2)

            if r_bbox__best is not None:
                img_best_boxes = cv2.circle(img_best_boxes, (int(cnt_best[0]), int(cnt_best[1])), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.line(img_best_boxes, tuple(r_bbox__best[0]), tuple(r_bbox__best[1]), (255, 0, 0), 3)
                cv2.line(img_best_boxes, tuple(r_bbox__best[1]), tuple(r_bbox__best[2]), (0, 0, 255), 3)
                cv2.line(img_best_boxes, tuple(r_bbox__best[2]), tuple(r_bbox__best[3]), (255, 0, 0), 3)
                cv2.line(img_best_boxes, tuple(r_bbox__best[3]), tuple(r_bbox__best[0]), (0, 0, 255), 3)

                ret_val = calc_jacc_IOU(gt_boxes_, gt_theta, r_bbox__best, theta_best)
                correct_boxes = correct_boxes + ret_val

    # visualization of results
    res = np.hstack((img, img_gt_boxes, img_all_boxes, img_best_boxes))
    scale_percent = 75  # % of original size
    width = int(res.shape[1] * scale_percent / 100)
    height = int(res.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, resized)

    return correct_boxes, total_boxes

# (pred, abs_paths, root_paths, im_size)
def save_prediction_image_OCID(raw_pred, img_abs_path, img_root_path, im_size, out_dir):
    num_classes_theta = 18
    threshold = 0.01
    total_boxes = 0
    correct_boxes = 0
    total_IOU_seg = 0
    IOU_count = 0
    IOU_seg_threshold = 100 # in px

    for i, (sem_pred, bbx_pred, cls_pred, obj_pred, iou_pred) in enumerate(zip(
            raw_pred["sem_pred"], raw_pred["bbx_pred"], raw_pred["cls_pred"], raw_pred["obj_pred"], raw_pred["iou_pred"])):

        item = os.path.join(img_root_path[i], img_abs_path[i])
        im_size_ = im_size[i]
        ensure_dir(out_dir)

        seq_path, im_name = item.split(',')
        sem_pred = np.asarray(sem_pred.detach().cpu().numpy(), dtype=np.uint8)
        seg_mask_vis = np.zeros((im_size_[0], im_size_[1], 3))
        cls_labels = np.unique(sem_pred)

        img_path = os.path.join(img_root_path[i], seq_path, 'rgb', im_name)
        mask_path = os.path.join(img_root_path[i], seq_path, 'seg_mask_labeled_combi', im_name)
        img = cv2.imread(img_path)
        img_best_boxes = np.copy(img)
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        delta_xy = np.array([[int(img.shape[1] / 2 - int(im_size_[1] / 2))],
                             [int(img.shape[0] / 2 - int(im_size_[0] / 2))]])

        # img_cls = np.copy(img)
        img_all_boxes = np.copy(img)
        img_IOU_seg = 0
        img_IOU_count = 0

        for cnt, label in enumerate(cls_labels):
            if label == 0:
                continue

            seg_mask_vis[sem_pred == label] = colors_list[label]
            mask_per_label = np.zeros_like(sem_pred)
            mask_per_label_gt = np.zeros_like(sem_pred)

            mask_per_label[sem_pred == label] = 1
            mask_per_label_gt[mask_gt == label] = 1

            if sum(map(sum, mask_per_label)) < IOU_seg_threshold:
                continue

            intersection = np.logical_and(mask_per_label_gt, mask_per_label)
            union = np.logical_or(mask_per_label_gt, mask_per_label)
            iou_score = np.sum(intersection) / np.sum(union)
            img_IOU_seg += iou_score
            img_IOU_count += 1

            # # only for object detection, to draw axis aligned bounding boxes
            # props = regionprops(mask_per_label)
            # for prop in props:
            #     cv2.rectangle(img_cls, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]),
            #                   colors_list[label].tolist(), 2)
            #     cv2.putText(img_cls, cls_list[label], (prop.bbox[1], prop.bbox[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 colors_list[label].tolist(), 1)

        total_IOU_seg += img_IOU_seg/img_IOU_count
        IOU_count += 1

        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, im_name[:-4] + ".png")

        img_mask = (img * 0.25 + seg_mask_vis * 0.75)

        anno_per_class_dir = os.path.join(os.path.join(img_root_path[i], seq_path, 'Annotations_per_class',
                                                       im_name[:-4]))

        for class_dir in os.listdir(anno_per_class_dir):
            if not os.path.isdir(os.path.join(anno_per_class_dir, class_dir)):
                continue

            gt_boxes_path = os.path.join(anno_per_class_dir, class_dir,im_name[:-4] + '.txt')
            gt_boxes = read_boxes_from_file(gt_boxes_path, delta_xy)

            if len(gt_boxes) == 0:
                print('txt file empty')
                continue
            else:
                total_boxes += 1

                if bbx_pred is None:
                    continue
                else:
                    (gt_boxes_, gt_theta, cls) = prepare_frcnn_format(gt_boxes, im_size_)

                    best_confidence = 0.
                    r_bbox__best = None
                    theta_best = None
                    cnt_best = None

                    for bbx_pred_i, cls_pred_i, obj_pred_i, iou_pred_i in zip(bbx_pred, cls_pred, obj_pred, iou_pred):

                        obj_iou = 0.3*iou_pred_i.item() + 0.7*obj_pred_i.item()
                        
                        if obj_iou > threshold:
                            pt1 = (int(bbx_pred_i[0]), int(bbx_pred_i[1]))
                            pt2 = (int(bbx_pred_i[2]), int(bbx_pred_i[3]))
                            cls = cls_pred_i.item()

                            theta = ((180 / num_classes_theta) * cls) + 5
                            pts = scipy.array([[pt1[0], pt1[1]], [pt2[0], pt1[1]], [pt2[0], pt2[1]], [pt1[0], pt2[1]]])
                            cnt = scipy.array([(int(bbx_pred_i[0]) + int(bbx_pred_i[2])) / 2,
                                               (int(bbx_pred_i[1]) + int(bbx_pred_i[3])) / 2])
                            r_bbox_ = Rotate2D(pts, cnt, 90 - theta)
                            r_bbox_ = r_bbox_.astype('int16')

                            if (int(cnt[1]) >= im_size_[0]) or (int(cnt[0]) >= im_size_[1]):
                                continue

                            if sem_pred[int(cnt[1]), int(cnt[0])] == int(class_dir):
                                if obj_iou >= best_confidence:
                                    best_confidence = obj_iou
                                    r_bbox__best = r_bbox_
                                    theta_best = theta
                                    cnt_best = cnt

                                cv2.line(img_all_boxes, tuple(r_bbox_[0]), tuple(r_bbox_[1]), (255, 0, 0), 2)
                                cv2.line(img_all_boxes, tuple(r_bbox_[1]), tuple(r_bbox_[2]), (0, 0, 255), 2)
                                cv2.line(img_all_boxes, tuple(r_bbox_[2]), tuple(r_bbox_[3]), (255, 0, 0), 2)
                                cv2.line(img_all_boxes, tuple(r_bbox_[3]), tuple(r_bbox_[0]), (0, 0, 255), 2)

                if r_bbox__best is not None:
                    img_best_boxes = cv2.circle(img_best_boxes, (int(cnt_best[0]), int(cnt_best[1])), radius=5, color=(0, 255, 0), thickness=-1)
                    cv2.line(img_best_boxes, tuple(r_bbox__best[0]), tuple(r_bbox__best[1]), (255, 0, 0), 3)
                    cv2.line(img_best_boxes, tuple(r_bbox__best[1]), tuple(r_bbox__best[2]), (0, 0, 255), 3)
                    cv2.line(img_best_boxes, tuple(r_bbox__best[2]), tuple(r_bbox__best[3]), (255, 0, 0), 3)
                    cv2.line(img_best_boxes, tuple(r_bbox__best[3]), tuple(r_bbox__best[0]), (0, 0, 255), 3)

                    ret_val = calc_jacc_IOU(gt_boxes_, gt_theta, r_bbox__best, theta_best)
                    correct_boxes = correct_boxes + ret_val

    # visualization of results
    res = np.hstack((img, img_all_boxes, img_best_boxes, img_mask))
    scale_percent = 75  # % of original size
    width = int(res.shape[1] * scale_percent / 100)
    height = int(res.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, resized)


    if IOU_count == 0:
        return correct_boxes, total_boxes, 0
    else:
        total_IOU_seg_mean = total_IOU_seg / IOU_count
        return correct_boxes, total_boxes, total_IOU_seg_mean


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_config(args):
    log_debug("Loading configuration from %s", args.config)
    conf = load_config(args.config, args.config)
    log_debug("\n%s", config_to_string(conf))
    return conf


def make_dataloader(args, config, rank, world_size):
    if args.dataset_name == 'OCID':
        config = config["dataloader_OCID"]
        log_debug("Creating dataloaders for dataset in %s", args.data)

        # Validation dataloader
        val_tf = OCIDTestTransform(config.getint("shortest_size"),
                                config.getint("longest_max_size"),
                                config.getstruct("rgb_mean"),
                                config.getstruct("rgb_std")
                                )
        val_db = OCIDTestDataset(args.data, config["root_path"], config["test_set"], val_tf)
        val_sampler = DistributedARBatchSampler(
            val_db, config.getint("val_batch_size"), world_size, rank, False)
        val_dl = data.DataLoader(val_db,
                                batch_sampler=val_sampler,
                                collate_fn=iss_collate_fn,
                                pin_memory=True,
                                num_workers=config.getint("num_workers"))
        return val_dl

    elif args.dataset_name == 'Cornell':
        config = config["dataloader_Cornell"]
        log_debug("Creating dataloaders for dataset in %s", args.data)

        # Validation dataloader
        val_tf = OCIDTestTransform(config.getint("shortest_size"),
                                config.getint("longest_max_size"),
                                config.getstruct("rgb_mean"),
                                config.getstruct("rgb_std")
                                )
        val_db = CornellDataset(args.data, config["root_path"], config["test_set"], val_tf)
        val_sampler = DistributedARBatchSampler(
            val_db, config.getint("val_batch_size"), world_size, rank, False)
        val_dl = data.DataLoader(val_db,
                                batch_sampler=val_sampler,
                                collate_fn=iss_collate_fn,
                                pin_memory=True,
                                num_workers=config.getint("num_workers"))
        return val_dl


def make_model(args, config):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    iou_config = config["iou"]
    general_config = config["general"]
    classes = {"total": int(general_config["num_things"]) + int(general_config["num_stuff"]), "stuff":
        int(general_config["num_stuff"]), "thing": int(general_config["num_things"]),
               "semantic": int(general_config["num_semantic"])}
    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)
    norm_act_static_iou, norm_act_dynamic_iou = norm_act_from_config(iou_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)
    if body_config.get("weights"):
        body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    # Freeze parameters
    for n, m in body.named_modules():
        for mod_id in range(1, body_config.getint("num_frozen") + 1):
            if ("mod%d" % mod_id) in n:
                freeze_params(m)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")
    fpn = FPN([body_channels[inp] for inp in fpn_inputs],
              fpn_config.getint("out_channels"),
              fpn_config.getint("extra_scales"),
              norm_act_static,
              fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator, anchor_matcher, rpn_loss,
        rpn_config.getint("anchor_scale"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create detection network

    prediction_generator = PredictionGenerator(roi_config.getfloat("nms_threshold"),
                                               roi_config.getfloat("score_threshold"),
                                               roi_config.getint("max_predictions"))
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))

    roi_loss = DetectionLoss(roi_config.getfloat("sigma"))

    roi_size = roi_config.getstruct("roi_size")
    roi_algo = DetectionAlgoFPN(
        prediction_generator, proposal_matcher, roi_loss, classes, roi_config.getstruct("bbx_reg_weights"),
        roi_config.getint("fpn_canonical_scale"), roi_config.getint("fpn_canonical_level"), roi_size,
        roi_config.getint("fpn_min_level"), roi_config.getint("fpn_levels"))

    roi_head = FPNROIHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    if args.dataset_name == 'OCID':
        # Create semantic segmentation network
        sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
        sem_algo = SemanticSegAlgo(sem_loss, classes["semantic"])
        sem_head = FPNSemanticHeadDeeplab(fpn_config.getint("out_channels"),
                                        sem_config.getint("fpn_min_level"),
                                        sem_config.getint("fpn_levels"),
                                        classes["semantic"],
                                        pooling_size=sem_config.getstruct("pooling_size"),
                                        norm_act=norm_act_static)

        fusion_module = FusionModule(32, 256, sem_config.getint("fpn_min_level"), sem_config.getint("fpn_levels"))
        return DetSegNet_OCID(body, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, fusion_module, classes)

    elif args.dataset_name == 'Cornell':
        return DetSegNet_Cornell(body, rpn_head, roi_head, rpn_algo, roi_algo)


def evaluate(args, model, dataloader, **varargs):
    model.eval()
    dataloader.batch_sampler.set_epoch(0)
    all_boxes = 0
    all_correct = 0
    total_IOU_seg = 0

    for it, batch in enumerate(dataloader):
        print('Batch no. : ' + str(it))
        with torch.no_grad():
            # Extract data
            img = batch["img"].cuda(device=varargs["device"], non_blocking=True)
            abs_paths = batch["abs_path"]
            root_paths = batch["root_path"]
            im_size = batch["im_size"]
            img_raw = batch["img_raw"]

            # Run network
            if args.dataset_name == 'OCID':
                _, pred, conf = model(img=img, do_loss=False, do_prediction=True)

                correct_boxes, total_boxes, total_IOU_seg_mean = varargs["save_function"](pred, abs_paths, root_paths, im_size)
                total_IOU_seg += total_IOU_seg_mean
                all_correct += correct_boxes
                all_boxes += total_boxes
                print('all boxes = ' + str(all_boxes))
                print('correct boxes = ' + str(all_correct))
                print('Mean Jaccard index = ' + str((all_correct / all_boxes)))
                print('-----------------------------------------------------')

            elif args.dataset_name == 'Cornell':
                _, pred= model(img=img, do_loss=False, do_prediction=True)

                correct_boxes, total_boxes= varargs["save_function"](img_raw, pred, abs_paths, root_paths, im_size)
                all_correct += correct_boxes
                all_boxes += total_boxes
                print('all boxes = ' + str(all_boxes))
                print('correct boxes = ' + str(correct_boxes))
                print('all correct boxes = ' + str(all_correct))
                print('Mean Jaccard index = ' + str((all_correct / all_boxes)))
                print('-----------------------------------------------------')

    if args.dataset_name == 'OCID':
        print('Total segmentation IOU = ' + str(total_IOU_seg/len(dataloader)))
        print('Total Jaccard index = ' + str(all_correct/all_boxes))
    elif args.dataset_name == 'Cornell':
        print('Total Jaccard index = ' + str(all_correct/all_boxes))


def main(args):
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Load configuration
    config = make_config(args)

    # Create dataloaders
    test_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(args, config)

    if args.dataset_name == 'OCID':
        snapshot = resume_from_snapshot(model, args.model, ["body", "rpn_head", "roi_head", "sem_head", "fusion_module"])
    elif args.dataset_name == 'Cornell':
        snapshot = resume_from_snapshot(model, args.model, ["body", "rpn_head", "roi_head"])
    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id,
                                    find_unused_parameters=True)
    if args.dataset_name == 'OCID':
        save_function = partial(save_prediction_image_OCID, out_dir=args.out_dir)
    elif args.dataset_name == 'Cornell':
        save_function = partial(save_prediction_image_Cornell, out_dir=args.out_dir)

    evaluate(args, model, test_dataloader, device=device, summary=None,
         log_interval=config["general"].getint("log_interval"), save_function=save_function)


if __name__ == "__main__":
    main(parser.parse_args())
