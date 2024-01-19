import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy

class Solver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(Solver, self).__init__(net, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        self.clustering_target_name = 'clustering_' + self.target_name
        assert('categorical' in self.train_data)

        num_layers = len(self.net.module.FC) + 1
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                  num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES, 
                  intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS, 
                                        self.opt.CLUSTERING.FEAT_KEY, 
                                        self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}
        
        self.decay1 =self.opt.CPD.DECAY1
        self.ce_loss, self.cdd_loss, self.pse_loss = self.opt.CPD.HYPER
    
    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        # if 'target_centers' not in self.history or \
        #         'ts_center_dist' not in self.history or \
        #         'target_labels' not in self.history:
        #     return False

        # if len(self.history['target_centers']) < 2 or \
		# len(self.history['ts_center_dist']) < 1 or \
		# len(self.history['target_labels']) < 2:
        #    return False

        # # target centers along training
        # target_centers = self.history['target_centers']
        # eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1], 
		# 	target_centers[-2])).item()

        # # target-source center distances along training
        # eval2 = self.history['ts_center_dist'][-1].item()

        # # target labels along training
        # path2label_hist = self.history['target_labels']
        # paths = self.clustered_target_samples['data']
        # num = 0
        # for path in paths:
        #     pre_label = path2label_hist[-2][path]
        #     cur_label = path2label_hist[-1][path]
        #     if pre_label != cur_label:
        #         num += 1
        # eval3 = 1.0 * num / len(paths)

        # return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
        #         eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
        #         eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        self.max_acc = 0.0
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True: 
            # updating the target label hypothesis through clustering
            self.target_prototypes = self.init_prototype()
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                #self.update_ss_alignment_loss_weight()
                print('Clustering based on target prototype')
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_centers = self.clustering.centers 
                center_change = self.clustering.center_change 
                path2label = self.clustering.path2label
                self.path2label = path2label
                # updating the history
                self.register_history('target_centers', target_centers,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
	            	self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                              self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'], 
                                                self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break
                
                # filtering the clustering results
                target_hypt, filtered_classes = self.filtering()

                # update dataloaders
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                # update train data setting
                self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            self.update_network(filtered_classes)
            self.loop += 1

        save_path = self.opt.SAVE_DIR
        acc = str(round(self.max_acc,1))
        out = save_path.split('/')
        out.pop()
        newout = ''
        for m in out:
            newout = os.path.join(newout,m)
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        newout = os.path.join(newout,acc+'_'+str(nowTime))
        os.rename(save_path, newout)
        print('Training Done!')
        
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        # source_dataloader = self.train_data[self.clustering_source_name]['loader']
        # net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        # source_centers = solver_utils.get_centers(net, 
		# source_dataloader, self.opt.DATASET.NUM_CLASSES, 
        #         self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = self.target_prototypes

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain()

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        chosen_samples = solver_utils.filter_samples(
		target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
		chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        if chosen_samples is not None and \
                              chosen_samples['gt'] is not None:
                    preds = to_onehot(chosen_samples['label'], 
                                                self.opt.DATASET.NUM_CLASSES)
                    gts = chosen_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('chosen_samples accuracy %s: %.4f' % (self.opt.EVAL_METRIC, res))
        self.selected_path2label = {}
        number_selected_samples = len(chosen_samples['data'])
        for i in range(number_selected_samples):
            self.selected_path2label[chosen_samples['data'][i]]=chosen_samples['label'][i].item()
        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = solver_utils.split_samples_classwise(
			samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                      for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]
        
        target_sample_labels = samples['Label_target']
        self.selected_classes = [labels[0].item() for labels in target_sample_labels]
        return target_samples, target_nums, target_sample_labels
            
    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.target_name]['iterator'] = \
                     iter(self.train_data[self.target_name]['loader'])
        self.train_data['categorical']['iterator'] = \
                     iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            pse_loss_iter = 0
            cdd_loss_iter = 0

            # coventional sampling 
            target_sample = self.get_samples(self.target_name) 
            target_data = target_sample['Img']
            target_path = target_sample['Path']
            target_data = to_cuda(target_data)
            self.net.module.set_bn_domain()
            result = self.net(target_data)
            target_preds = result['logits']
            target_feats = result['feat']

            ### update target prototype 
            
            pre = nn.Softmax(dim=1)(target_preds)
            target_prototypes_temp = target_feats.t().mm(pre).t()
            self.target_prototypes = self.decay1 * self.target_prototypes + (1 - self.decay1) * target_prototypes_temp
            

            # compute the cross-entropy loss of target prototypes
            logit_target_prototypes = self.net.module.get_logit(self.target_prototypes)
            label_target_prototypes = to_cuda(torch.arange(self.opt.DATASET.NUM_CLASSES).long())
            ce_loss = self.CELoss(logit_target_prototypes, label_target_prototypes)
            ce_loss_iter += ce_loss
            loss += ce_loss
            # computer the cross-entropy loss of pseudo labeled target samples 
            clustered_label = [self.selected_path2label.get(path) for path in target_path]
            mask = [i !=None for i in clustered_label]
            clustered_label = [i for i in clustered_label if i!=None]
            target_preds = target_preds[mask]
            clustered_label = to_cuda(torch.Tensor(clustered_label).long())
            pse_loss = self.CELoss(target_preds, clustered_label)
            pse_loss_iter += pse_loss
            loss += pse_loss
         
            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                target_samples_cls, target_nums_cls, target_sample_labels  = self.CAS()

                # 2) forward and compute the loss
                target_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain()
                feats_target = self.net(target_cls_concat)

                # prepare the features                                                             
                feats_toalign_T = self.prepare_feats(feats_target)
                selected_classes = [labels[0].item() for labels in target_sample_labels]
                selected_target_prototype = self.target_prototypes[selected_classes].detach()
                selected_target_prototype_probs = nn.Softmax(dim=1)(self.net.module.get_logit(selected_target_prototype)).detach()
                feats_toalign_S = [selected_target_prototype,selected_target_prototype_probs]
                source_nums_cls = [1] * len(target_nums_cls)
                cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T, 
                               source_nums_cls, target_nums_cls)[self.discrepancy_key]
                
                # cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                total = self.ce_loss*ce_loss+self.cdd_loss*cdd_loss+self.pse_loss*pse_loss
                total.backward()
                self.target_prototypes = self.target_prototypes.detach()

                cdd_loss_iter += cdd_loss
                loss += cdd_loss

            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                cur_loss = {'ce_loss': ce_loss_iter, 'cdd_loss': cdd_loss_iter,
			'total_loss': loss}
                self.logging(cur_loss)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain()
                    self.temp_accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, self.temp_accu))
                    a = self.max_acc if self.max_acc>self.temp_accu else self.temp_accu
                    print('max acc:' + str(a))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                if self.temp_accu > self.max_acc:
                    self.max_acc = self.temp_accu
                    self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

    def init_prototype(self):
        with torch.no_grad():
            self.net.eval()
            fea = torch.Tensor([])
            p = torch.Tensor([])
            for sample in iter(self.test_data['loader']):
                input_t = to_cuda(sample['Img'])
                fea_temp = self.net(input_t)['feat'].cpu()
                p_temp = self.net(input_t)['logits'].cpu()
                fea = torch.cat((fea, fea_temp), dim=0)
                p = torch.cat((p, p_temp), dim=0)
            p = nn.Softmax(dim=1)(p)
            f_t = to_cuda(fea.t().mm(p).t())
        return f_t

    def logging(self, loss):
        print('[loop: %d, iters: %d]: ' % (self.loop, self.iters))
        loss_names = ""
        loss_values = ""
        for key in loss:
            loss_names += key + ","
            loss_values += '%.4f,' % (loss[key])
        loss_names = loss_names[:-1] + ': '
        loss_values = loss_values[:-1] + ';'
        loss_str = loss_names + loss_values 
        print(loss_str)

