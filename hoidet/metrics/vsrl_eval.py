# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta 
# 
# This file is part of the VCOCO dataset hooks and is available 
# under the terms of the Simplified BSD License provided in 
# LICENSE. Please retain this notice and LICENSE if you use 
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
from pycocotools.coco import COCO
import json
import copy
import pickle
from tqdm import tqdm

class VCOCOeval(object):

    def __init__(self, vsrl_annot_file, coco_annot_file,
                 split_file):
        """Input:
        vslr_annot_file: path to the vcoco annotations
        coco_annot_file: path to the coco annotations
        split_file: image ids for split
        """
        self.COCO = COCO(coco_annot_file)  # 所有V-COCO图片标注数据
        self.VCOCO = _load_vcoco(vsrl_annot_file)
        self.image_ids = np.loadtxt(open(split_file, 'r'))
        # simple check
        assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), self.image_ids))

        # TODO: 发现这里存储了很多完全一样的字段，待优化
        for i in range(len(self.VCOCO)):
            assert np.all(np.equal(self.VCOCO[0]['image_id'], self.VCOCO[i]['image_id']))
            assert np.all(np.equal(self.VCOCO[0]['ann_id'], self.VCOCO[i]['ann_id']))
            assert np.all(np.equal(self.VCOCO[0]['image_id'], self.VCOCO[i]['image_id']))
            assert np.max(self.VCOCO[i]['label']) == 1

        self._init_coco()
        self._init_vcoco()

    def _init_vcoco(self):
        actions = [x['action_name'] for x in self.VCOCO]  # 26 个动作名称列表
        roles = [x['role_name'] for x in self.VCOCO]  # 26 个动作对应的角色
        self.actions = actions
        self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
        self.num_actions = len(self.actions)  # 26
        self.roles = roles

    def _init_coco(self):
        category_ids = self.COCO.getCatIds()  # 80 个物体类别编号，范围是[1, 90]
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]  # 80 个物体类别文本标签
        self.category_to_id_map = dict(zip(categories, category_ids))  # 文本到类别编号
        self.classes = ['__background__'] + categories  # 添加一个背景类别
        self.num_classes = len(self.classes)  # 81
        self.json_category_id_to_contiguous_id = {  # 将COCO的 80 个类别标签从 [1,90] 映射到 [1, 80]。0 保留给背景类别
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {  # 从 [1,80] 映射回 [1,90] 范围的 coco 标签
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}

    def _get_vcocodb(self):
        vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))
        for entry in vcocodb:
            self._prep_vcocodb_entry(entry)
            self._add_gt_annotations(entry)

            # 移除不需要的字段
            entry.pop("license")
            entry.pop("file_name")
            entry.pop("coco_url")
            entry.pop("date_captured")
            entry.pop("flickr_url")
            entry.pop("is_crowd")

        # print
        if 0:
            nums = np.zeros((self.num_actions), dtype=np.int32)
            for entry in vcocodb:
                for aid in range(self.num_actions):
                    nums[aid] += np.sum(np.logical_and(entry['gt_actions'][:, aid] == 1, entry['gt_classes'] == 1))
            for aid in range(self.num_actions):
                print('Action %s = %d' % (self.actions[aid], nums[aid]))

        return vcocodb

    def _prep_vcocodb_entry(self, entry):
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['is_crowd'] = np.empty((0), dtype=np.bool_)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
        entry['gt_role_id'] = np.empty((0, self.num_actions, 2), dtype=np.int32)

    def _add_gt_annotations(self, entry):
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []  # valid_objs[i]是第i个box
        valid_ann_ids = []  # valid_ann_ids[i]是第i个box的标注信息编号
        width = entry['width']
        height = entry['height']
        for i, obj in enumerate(objs):
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
            x1, y1, x2, y2 = clip_xyxy_to_image(
                x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_ann_ids.append(ann_ids[i])
        num_valid_objs = len(valid_objs)
        assert num_valid_objs == len(valid_ann_ids)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
        gt_role_id = -np.ones((num_valid_objs, self.num_actions, 2), dtype=entry['gt_role_id'].dtype)

        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            is_crowd[ix] = obj['iscrowd']

            # gt_actions[ix]是第ix个box的动作类别，采用one-hot编码，1表示存在该动作，可能会存在多个动作
            # gt_role_id[ix]是第ix个box的作用物box在COCO数据集中的注释编号
            gt_actions[ix, :], gt_role_id[ix, :, :] = \
                self._get_vsrl_data(valid_ann_ids[ix],
                                    valid_ann_ids, valid_objs)

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
        entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)

    def _get_vsrl_data(self, ann_id, ann_ids, objs):
        """ Get VSRL data for ann_id."""
        action_id = -np.ones((self.num_actions), dtype=np.int32)
        role_id = -np.ones((self.num_actions, 2), dtype=np.int32)
        # check if ann_id in vcoco annotations
        in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
        if in_vcoco.size > 0:
            action_id[:] = 0
            role_id[:] = -1
        else:
            return action_id, role_id  # 全为-1表示该边界框内没有任何交互动作
        for i, x in enumerate(self.VCOCO):
            assert x['action_name'] == self.actions[i]
            has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
            if has_label.size > 0:
                action_id[i] = 1
                assert has_label.size == 1
                rids = x['role_object_id'][has_label]
                assert rids[0, 0] == ann_id
                for j in range(1, rids.shape[1]):
                    if rids[0, j] == 0:
                        # no role
                        continue
                    aid = np.where(ann_ids == rids[0, j])[0]
                    assert aid.size == 1
                    role_id[i, j - 1] = aid[0]
        return action_id, role_id

    def _collect_detections_for_image(self, dets, image_id):
        agents = np.empty((0, 4 + self.num_actions), dtype=np.float32)
        roles = np.empty((0, 5 * self.num_actions, 2), dtype=np.float32)
        for det in dets:
            if det['image_id'] == image_id:
                this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
                this_role = np.zeros((1, 5 * self.num_actions, 2), dtype=np.float32)
                this_agent[0, :4] = det['person_box']
                for aid in range(self.num_actions):
                    for j, rid in enumerate(self.roles[aid]):
                        if rid == 'agent':
                            this_agent[0, 4 + aid] = det[self.actions[aid] + '_' + rid]
                        else:
                            this_role[0, 5 * aid: 5 * aid + 5, j - 1] = det[self.actions[aid] + '_' + rid]
                agents = np.concatenate((agents, this_agent), axis=0)
                roles = np.concatenate((roles, this_role), axis=0)
        return agents, roles

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

    def _do_eval(self, detections_file, ovr_thresh=0.5):
        """弃用"""
        vcocodb = self._get_vcocodb()
        # self._do_agent_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh)

        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)
        self._pre_collect_detections_for_image(dets)

        self._do_role_eval(vcocodb, ovr_thresh=ovr_thresh, eval_type='scenario_1')
        self._do_role_eval(vcocodb, ovr_thresh=ovr_thresh, eval_type='scenario_2')

    def eval(self, detections_file, ovr_thresh=0.5):
        """
        同时计算Scenario 1和Scenario 2的role AP
        Args:
            detections_file: 模型预测结果
            ovr_thresh:

        Returns:

        """
        vcocodb = self._get_vcocodb()

        with open(detections_file, 'rb') as f:
            dets = pickle.load(f)
        self._pre_collect_detections_for_image(dets)

        self._do_role_eval_2(vcocodb, ovr_thresh=ovr_thresh)

    def _do_role_eval(self, vcocodb, ovr_thresh=0.5, eval_type='scenario_1'):

        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]

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

                    # keep track of detected instances for each action for each role
                    covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool_)

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
                        sc[aid][rid].append(agent_scores[j])
                        if not (is_true_action and ovmax >= ovr_thresh):
                            fp[aid][rid].append(1)
                            tp[aid][rid].append(0)
                            continue

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1):  # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false
                                    ov_role = 0.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0
                            else:
                                raise ValueError('Unknown eval type')
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

                        if ov_role >= ovr_thresh:
                            if covered[jmax]:
                                fp[aid][rid].append(1)
                                tp[aid][rid].append(0)
                            else:
                                fp[aid][rid].append(0)
                                tp[aid][rid].append(1)
                                covered[jmax] = True
                        else:
                            fp[aid][rid].append(1)
                            tp[aid][rid].append(0)

        role_ap = _compute_role_ap(self.num_actions, self.roles, fp, tp, sc, npos)

        # # compute ap for each action
        # role_ap = np.zeros((self.num_actions, 2), dtype=np.float32)
        # role_ap[:] = np.nan
        # for aid in range(self.num_actions):
        #     if len(self.roles[aid]) < 2:
        #         continue
        #     for rid in range(len(self.roles[aid]) - 1):
        #         a_fp = np.array(fp[aid][rid], dtype=np.float32)
        #         a_tp = np.array(tp[aid][rid], dtype=np.float32)
        #         a_sc = np.array(sc[aid][rid], dtype=np.float32)
        #         # sort in descending score order
        #         idx = a_sc.argsort()[::-1]
        #         a_fp = a_fp[idx]
        #         a_tp = a_tp[idx]
        #         a_sc = a_sc[idx]
        #
        #         a_fp = np.cumsum(a_fp)
        #         a_tp = np.cumsum(a_tp)
        #         rec = a_tp / float(npos[aid])
        #         # check
        #         assert (np.amax(rec) <= 1)
        #         prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
        #         role_ap[aid, rid] = voc_ap(rec, prec)

        print('---------Reporting Role AP (%)------------------')
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2: continue
            for rid in range(len(self.roles[aid]) - 1):
                print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid] + '-' + self.roles[aid][rid + 1],
                                                                   role_ap[aid, rid] * 100.0, int(npos[aid])))
        print('Average Role [%s] AP = %.2f' % (eval_type, np.nanmean(role_ap) * 100.00))
        print('---------------------------------------------')
        return role_ap

    def _do_role_eval_2(self, vcocodb, ovr_thresh=0.5):

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

        # compute ap for each action
        role_ap_1 = _compute_role_ap(self.num_actions, self.roles, fp_1, tp_1, sc_1, npos)
        role_ap_2 = _compute_role_ap(self.num_actions, self.roles, fp_2, tp_2, sc_2, npos)

        _print_role_ap(
            num_actions=self.num_actions,
            roles=self.roles,
            actions=self.actions,
            npos=npos,
            role_ap=role_ap_1,
            eval_type=1
        )
        _print_role_ap(
            num_actions=self.num_actions,
            roles=self.roles,
            actions=self.actions,
            npos=npos,
            role_ap=role_ap_2,
            eval_type=2
        )

        return role_ap_1, role_ap_2


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



def _load_vcoco(vcoco_file):
    print('loading vcoco annotations...')
    with open(vcoco_file, 'r') as f:
        vsrl_data = json.load(f)
    for i in range(len(vsrl_data)):
        vsrl_data[i]['role_object_id'] = \
            np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
        for j in ['ann_id', 'label', 'image_id']:
            vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
    return vsrl_data


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


if __name__ == "__main__":
    vsrl_annot_file = "/workspace/tmp/v-coco/data/vcoco/vcoco_test.json"
    coco_file = "/workspace/tmp/v-coco/data/instances_vcoco_all_2014.json"
    split_file = "/workspace/tmp/v-coco/data/splits/vcoco_test.ids"

    # Change this line to match the path of your cached file
    det_file = "/workspace/code/dl_github/HOIDet/data/cache.pkl"

    # print("Loading cached results from {det_file}.")
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    # vcocoeval._do_eval(det_file, ovr_thresh=0.5)
    vcocoeval.eval(det_file, ovr_thresh=0.5)
