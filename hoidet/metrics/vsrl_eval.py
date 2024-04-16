"""
以下代码修改自：https://github.com/s-gupta/v-coco/blob/master/vsrl_eval.py
"""

# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta 
# 
# This file is part of the VCOCO dataset hooks and is available 
# under the terms of the Simplified BSD License provided in 
# LICENSE. Please retain this notice and LICENSE if you use 
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

import numpy as np
import pickle
from tqdm import tqdm


class VCOCOeval:
    def __init__(self, pred_file_path, evaluation_file_path):
        with open(pred_file_path, "rb") as fd:
            preds = pickle.load(fd)
        with open(evaluation_file_path, "rb") as fd:
            eval_target = pickle.load(fd)

        self.vcocodb = eval_target["vcocodb"]
        self.num_actions = eval_target["num_actions"]
        self.roles = eval_target["roles"]
        self.actions = eval_target["actions"]

        self._pre_collect_detections_for_image(preds)

    def _pre_collect_detections_for_image(self, dets):
        """预先收集好每张图片的检测结果，从而避免调用_collect_detections_for_image方法，以优化性能"""
        agents = dict()
        roles = dict()
        for det in tqdm(dets):
            image_id = det['image_id']
            this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
            this_role = np.zeros((1, 5 * self.num_actions, 2), dtype=np.float32)
            this_agent[0, :4] = det['person_box']
            for aid in range(self.num_actions):
                for j, rid in enumerate(self.roles[aid]):
                    if rid == 'agent':
                        this_agent[0, 4 + aid] = det[self.actions[aid] + '_' + rid]
                    else:
                        this_role[0, 5 * aid: 5 * aid + 5, j - 1] = det[self.actions[aid] + '_' + rid]

            if image_id not in agents.keys():
                assert image_id not in roles.keys()
                agents[image_id] = [this_agent]
                roles[image_id] = [this_role]
            else:
                agents[image_id].append(this_agent)
                roles[image_id].append(this_role)

        self._detection_agents_for_image = dict()
        self._detection_roles_for_image = dict()
        for key in agents.keys():
            self._detection_agents_for_image[key] = np.concatenate(agents[key], axis=0)
            self._detection_roles_for_image[key] = np.concatenate(roles[key], axis=0)

        # 验证是否正确
        # print("Checking...")
        # for key in tqdm(agents.keys()):
        #     pred_agents, pred_roles = self._collect_detections_for_image(dets, key)
        #     assert np.all(pred_agents == self._detection_agents_for_image[key])
        #     assert np.all(pred_roles == self._detection_roles_for_image[key])
        # print("Check: OK")

    def eval(self, ovr_thresh=0.5):
        return self._do_role_eval(self.vcocodb, ovr_thresh=ovr_thresh)

    def _do_role_eval(self, vcocodb, ovr_thresh=0.5):

        # scenario_1
        tp_1 = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp_1 = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc_1 = [[[] for r in range(2)] for a in range(self.num_actions)]
        # scenario_2
        tp_2 = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp_2 = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc_2 = [[[] for r in range(2)] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)  # npos[i] 表示第 i 个动作有多少个实例

        for i in tqdm(range(len(vcocodb))):
            image_id = vcocodb[i]['id']
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]  # person的标签编号是1，这里是为了得到人框的索引
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]  # 人框
            gt_actions = vcocodb[i]['gt_actions'][gt_inds]  # 人框对应的动作类别（一个人框可能有多个动作）, -1 表示没有任何动作
            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(gt_actions[:, aid] == 1)  # 统计每个action的实例个数

            # pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)
            if image_id not in self._detection_agents_for_image.keys():
                pred_agents = None
                pred_roles = None
            else:
                pred_agents = self._detection_agents_for_image[image_id]
                pred_roles = self._detection_roles_for_image[image_id]
            if pred_roles is None:
                # 如果模型在该图片上的预测结果为空，则没必要再执行如下代码
                continue

            for aid in range(self.num_actions):
                if len(self.roles[aid]) < 2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid]) - 1):

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid, rid]
                    gt_roles = -np.ones_like(gt_boxes)
                    for j in range(gt_boxes.shape[0]):
                        if gt_role_inds[j] > -1:
                            gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
                    agent_scores = pred_roles[:, 5 * aid + 4, rid]

                    valid = np.where(np.isnan(agent_scores) == False)[0]
                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    idx = agent_scores.argsort()[::-1]

                    # keep track of detected instances for each action for each role
                    covered_1 = np.zeros((gt_boxes.shape[0]), dtype=np.bool_)
                    covered_2 = np.zeros((gt_boxes.shape[0]), dtype=np.bool_)
                    for j in idx:
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_boxes, pred_box)

                        # matching happens based on the person
                        jmax = overlaps.argmax()
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        is_true_action = (gt_actions[jmax, aid] == 1)
                        sc_1[aid][rid].append(agent_scores[j])
                        sc_2[aid][rid].append(agent_scores[j])
                        if not (is_true_action and ovmax >= ovr_thresh):
                            fp_1[aid][rid].append(1)
                            tp_1[aid][rid].append(0)
                            fp_2[aid][rid].append(1)
                            tp_2[aid][rid].append(0)
                            continue

                        # overlap between predicted role and gt role
                        ov_role_1 = 0.0
                        ov_role_2 = 1.0
                        if np.all(gt_roles[jmax, :] == -1):
                            if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):  # if no gt role
                                ov_role_1 = 1.0
                        else:
                            ov_role_1 = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])
                            ov_role_2 = ov_role_1

                        # 场景1
                        if ov_role_1 >= ovr_thresh:
                            if covered_1[jmax]:
                                fp_1[aid][rid].append(1)
                                tp_1[aid][rid].append(0)
                            else:
                                fp_1[aid][rid].append(0)
                                tp_1[aid][rid].append(1)
                                covered_1[jmax] = True
                        else:
                            fp_1[aid][rid].append(1)
                            tp_1[aid][rid].append(0)

                        # 场景2
                        if ov_role_2 >= ovr_thresh:
                            if covered_2[jmax]:
                                fp_2[aid][rid].append(1)
                                tp_2[aid][rid].append(0)
                            else:
                                fp_2[aid][rid].append(0)
                                tp_2[aid][rid].append(1)
                                covered_2[jmax] = True
                        else:
                            fp_2[aid][rid].append(1)
                            tp_2[aid][rid].append(0)

        self.npos = npos

        # compute ap for each action
        role_ap_1 = _compute_role_ap(self.num_actions, self.roles, fp_1, tp_1, sc_1, npos)
        role_ap_2 = _compute_role_ap(self.num_actions, self.roles, fp_2, tp_2, sc_2, npos)

        return role_ap_1, role_ap_2

    def print_role_ap(self, role_ap, eval_type):
        _print_role_ap(
            num_actions=self.num_actions,
            roles=self.roles,
            actions=self.actions,
            npos=self.npos,
            role_ap=role_ap,
            eval_type=eval_type
        )


def _print_role_ap(num_actions, roles, actions, npos, role_ap, eval_type):
    print('---------Reporting Role AP (%)------------------')
    for aid in range(num_actions):
        if len(roles[aid]) < 2: continue
        for rid in range(len(roles[aid]) - 1):
            print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(actions[aid] + '-' + roles[aid][rid + 1],
                                                               role_ap[aid, rid] * 100.0, int(npos[aid])))
    eval_type = f"scenario_{eval_type}"
    print(f'Average Role [%s] AP = %.2f' % (eval_type, np.nanmean(role_ap) * 100.00))
    print('---------------------------------------------')


def _compute_role_ap(num_actions, roles, fp, tp, sc, npos):
    # compute ap for each action
    role_ap = np.zeros((num_actions, 2), dtype=np.float32)
    role_ap[:] = np.nan
    for aid in range(num_actions):
        if len(roles[aid]) < 2:
            continue
        for rid in range(len(roles[aid]) - 1):
            a_fp = np.array(fp[aid][rid], dtype=np.float32)
            a_tp = np.array(tp[aid][rid], dtype=np.float32)
            a_sc = np.array(sc[aid][rid], dtype=np.float32)
            # sort in descending score order
            idx = a_sc.argsort()[::-1]
            a_fp = a_fp[idx]
            a_tp = a_tp[idx]

            a_fp = np.cumsum(a_fp)
            a_tp = np.cumsum(a_tp)
            rec = a_tp / float(npos[aid])
            # check
            assert (np.amax(rec) <= 1)
            prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
            role_ap[aid, rid] = voc_ap(rec, prec)

    return role_ap


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def get_overlap(boxes, ref_box):
    ixmin = np.maximum(boxes[:, 0], ref_box[0])
    iymin = np.maximum(boxes[:, 1], ref_box[1])
    ixmax = np.minimum(boxes[:, 2], ref_box[2])
    iymax = np.minimum(boxes[:, 3], ref_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
           (boxes[:, 2] - boxes[:, 0] + 1.) *
           (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    [as defined in PASCAL VOC]
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

