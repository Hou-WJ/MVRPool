import os

import time
import random
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchinfo
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader, DenseDataLoader
from ogb.graphproppred import Evaluator
from data_utils import build_viewlist_from_args, load_dataset, max_node_nums
from models.MVRPool import MVRPool

from utils import set_gpu, make_directory
from config import args


class Trainer(object):
    def __init__(self, params):
        self.args = params

        # set GPU
        if self.args.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.args.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.args.use_node_attr = True
        self.args.rewired_data = True if self.args.model in ["MVRPool", "RGIN", "RSAGE", "RGAT", "RGCN", "MVPool_draw"] else False
        self.data = None
        self.load_data()

        # build the model
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

    # load data
    def load_data(self):
        print("Use node attr:", self.args.use_node_attr)
        print("Node attr norm:", self.args.data_normalization)
        print("Use rewired data:", self.args.rewired_data)

        if self.args.rewired_data:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'rewired_data', self.args.dataset)
            viewlist = build_viewlist_from_args(self.args)
        else:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', self.args.dataset)
            viewlist = None

        # print(viewlist)
        dataset, task_type = load_dataset(dataset_name=self.args.dataset, path=path, viewlist=viewlist,
                                       data_normalization=self.args.data_normalization)
        # dataset.data.edge_attr = None

        if not self.args.dataset.startswith('ogbg-'):
            dataset.data.edge_attr = None
        else:
            self.evaluator = Evaluator(name=self.args.dataset)

        self.data = dataset
        self.task_type = task_type
        self.args.task_type = task_type

        self.data.max_node_nums = max_node_nums(self.data)
        self.data.viewlist = viewlist

        if self.args.dataset.startswith('ogbg-'):
            self.data.target_dim = self.data.num_tasks
            print(f"OGB Tasks DIM: {self.data.target_dim}, Task Type: {dataset.task_type}")
        elif self.task_type == "regression":
            self.data.target_dim = 1 if self.data.y.dim() == 1 else self.data.y.size(1)
            print("Regression DIM:", self.data.target_dim)
        else:
            self.data.target_dim = self.data.num_classes

        if self.args.rewired_data:
            self.args.datasetID = self.data.hash_id

        print("Dataset:", self.data.data)
        print("Graph num:", self.data.len())
        print("Max node num:", self.data.max_node_nums)
        print("View List:", self.data.viewlist)

    # load model
    def add_model(self):
        if self.args.model == '':
            model = None
        elif self.args.model == "MVRPool":
            model = MVRPool(self.data, self.args)
        else:
            raise NotImplementedError
        # model.to(self.device).reset_parameters()
        model.to(self.device)
        return model

    def add_optimizer(self):
        # return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def add_lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step,
                                               gamma=self.args.lr_decay_factor)

    # save model locally
    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'args': vars(self.args)
        }
        # print(save_path)
        torch.save(state, save_path)

    # load model from path
    def load_model(self, load_path):
        state = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

    # use 10-fold cross-validation
    def k_fold(self):
        kf = KFold(self.args.folds, shuffle=True, random_state=self.args.seed)

        test_indices, train_indices = [], []
        for _, idx in kf.split(torch.zeros(len(self.data)), self.data.data.y):
            test_indices.append(torch.from_numpy(idx))


        val_indices = [test_indices[i - 1] for i in range(self.args.folds)]
        for i in range(self.args.folds):
            train_mask = torch.ones(len(self.data), dtype=torch.uint8)
            ##
            test_indices[i] = test_indices[i].type(torch.long)
            val_indices[i] = val_indices[i].type(torch.long)
            ##
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))

        return train_indices, test_indices, val_indices

    def ogb_split(self):
        split_idx = self.data.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"]
        return train_idx, val_idx, test_idx

    def _unpack_output(self, model_output):
        if isinstance(model_output, tuple):
            if len(model_output) == 2:
                return model_output[0], model_output[1]
            elif len(model_output) == 3:
                return model_output[0], model_output[1]
            else:
                raise ValueError(f"Model returned {len(model_output)} values, expected 2 or 3.")
        else:
            return model_output, 0

    def run_epoch(self, loader):
        self.model.train()

        loss_train = 0
        correct = 0
        mae = 0
        extra_loss = 0
        cls_loss = 0
        total_samples = 0
        for data in loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)

            out = self.model(data)
            out, d = self._unpack_output(out)
            data_num = data.num_graphs

            if self.args.dataset.startswith('ogbg-'):
                if out.shape[-1] == 2 and data.y.shape[-1] == 1:
                    # 或者使用 out = out.argmax(dim=-1, keepdim=True).float()，
                    # 但既然用 BCEWithLogitsLoss，通常模型最后一层输出应该代表 logits，我们取第二个通道（代表类别1的得分）
                    out_to_loss = out[:, 1:2]
                else:
                    out_to_loss = out

                    # 过滤掉 NaN 标签
                is_labeled = data.y == data.y

                if 'classification' in self.data.task_type:
                    # 使用调整过维度的 out_to_loss
                    loss = F.binary_cross_entropy_with_logits(out_to_loss[is_labeled],
                                                              data.y.to(torch.float)[is_labeled])
                else:
                    loss = F.mse_loss(out_to_loss[is_labeled], data.y.to(torch.float)[is_labeled])

                correct = 0
            else:
                if self.task_type == 'classification':
                    loss = F.nll_loss(out, data.y.view(-1))
                    pred = out.argmax(dim=1)
                    correct += pred.eq(data.y.view(-1)).sum().item()
                else:
                    loss = F.mse_loss(out, data.y.view(-1, self.data.target_dim).float())

                    # print("out loss", loss)
                    mae += torch.abs(out - data.y.view(-1, self.data.target_dim)).sum().item()

            cls_loss += loss * data_num

            loss = loss + d
            loss.backward()

            loss_train += loss.item() * data_num
            extra_loss += d * data_num
            total_samples += data_num

            self.optimizer.step()
            self.lr_scheduler.step()

        if self.task_type == 'classification':
            metric = correct / total_samples
        else:
            metric = mae / total_samples

        return loss_train / total_samples, cls_loss / total_samples, extra_loss / total_samples, metric

    # validate model
    def validate(self, loader):
        self.model.eval()

        total_loss = 0
        total_samples = 0
        correct = 0
        mae = 0

        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
                out, d = self._unpack_output(out)

            if self.args.dataset.startswith('ogbg-'):
                if out.shape[-1] == 2 and data.y.shape[-1] == 1:
                    out_to_loss = out[:, 1:2]
                else:
                    out_to_loss = out

                is_labeled = data.y == data.y

                if 'classification' in self.data.task_type:
                    loss = F.binary_cross_entropy_with_logits(out_to_loss[is_labeled],
                                                              data.y.to(torch.float)[is_labeled])
                else:
                    loss = F.mse_loss(out_to_loss[is_labeled], data.y.to(torch.float)[is_labeled])

                correct = 0
            else:
                if self.task_type == 'classification':
                    loss = F.nll_loss(out, data.y.view(-1), reduction='sum')
                    pred = out.argmax(dim=1)
                    correct += pred.eq(data.y.view(-1)).sum().item()
                else:
                    loss = F.mse_loss(out, data.y.view(-1, self.data.target_dim), reduction='sum')
                    mae += torch.abs(out - data.y.view(-1, self.data.target_dim)).sum().item()

            total_loss += loss.item()
            total_samples += data.y.size(0)

        avg_loss = total_loss / total_samples
        if self.task_type == 'classification':
            metric = correct / total_samples
        else:
            metric = mae / total_samples

        return avg_loss, metric

    # test model
    def predict(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        correct = 0

        all_preds = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _ = self._unpack_output(self.model(data))

            if self.args.dataset.startswith('ogbg-'):
                if out.shape[-1] == 2 and data.y.shape[-1] == 1:
                    out_to_loss = out[:, 1:2]
                else:
                    out_to_loss = out

                is_labeled = data.y == data.y

                if 'classification' in self.data.task_type:
                    loss = F.binary_cross_entropy_with_logits(out_to_loss[is_labeled],
                                                              data.y.to(torch.float)[is_labeled])
                else:
                    loss = F.mse_loss(out_to_loss[is_labeled], data.y.to(torch.float)[is_labeled])

                correct = 0
            else:

                if self.task_type == 'classification':
                    out = F.log_softmax(out, dim=-1)
                    loss = F.nll_loss(out, data.y.view(-1), reduction='sum')
                    pred = out.argmax(dim=1)
                    correct += pred.eq(data.y.view(-1)).sum().item()
                else:  # regression

                    loss = F.mse_loss(out, data.y.view(-1, self.data.target_dim), reduction='sum')  # self.criterion = nn.MSELoss()
                    pred = out.view(-1, self.data.target_dim)

            total_loss += loss.item()
            total_samples += data.num_graphs

            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        avg_loss = total_loss / total_samples
        if self.task_type == 'classification':
            # accuracy = (all_preds == all_labels).mean()
            accuracy = correct / total_samples
            cm = confusion_matrix(all_labels, all_preds)
            cr = classification_report(all_labels, all_preds, digits=4)
            return avg_loss, accuracy, cm, cr
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            return avg_loss, mse, mae, r2

    # main function for running the experiments
    def run(self):
        val_accs, test_accs, train_times = [], [], []
        make_directory('{}/{}/'.format(self.args.directory_path, self.args.log_db))
        best_val_model_save_path = '{}/{}/seed_{}_best_val_model.pth'.format(self.args.directory_path, self.args.log_db,
                                                                             self.args.counter + 1)
        best_test_model_save_path = '{}/{}/seed_{}_best_test_model.pth'.format(self.args.directory_path,
                                                                               self.args.log_db, self.args.counter + 1)
        run_test_result_path = '{}/{}/seed_{}_test_result.txt'.format(self.args.directory_path, self.args.log_db,
                                                                      self.args.counter + 1)
        run_test_acc_path = '{}/{}/seed_{}_test_loss_and_acc.csv'.format(self.args.directory_path, self.args.log_db,
                                                                         self.args.counter + 1)
        run_test_acc_list = []
        best_fold_test_acc = 0

        if self.args.restore:
            self.load_model(best_val_model_save_path)
            print('Successfully Loaded previous model')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.args.dataset.startswith('ogbg-'):

            train_idx, val_idx, test_idx = self.ogb_split()

            self.model = self.add_model()
            self.optimizer = self.add_optimizer()
            self.lr_scheduler = self.add_lr_scheduler()

            self.model_model = torch.compile(self.model)

            print("模型当前所在的设备:", next(self.model.parameters()).device)

            train_loader = DataLoader(self.data[train_idx], self.args.batch_size, shuffle=True)
            val_loader = DataLoader(self.data[val_idx], self.args.batch_size, shuffle=False)
            test_loader = DataLoader(self.data[test_idx], self.args.batch_size, shuffle=False)

            best_val_perf = -float('inf') if 'classification' in self.data.task_type else float('inf')
            best_test_perf = 0

            patience = self.args.patience
            patience_counter = 0
            best_model_state = None
            ogb_model_save_path = '{}/{}/seed_{}_ogb_best_model.pth'.format(
                self.args.directory_path, self.args.log_db, self.args.counter + 1
            )
            os.makedirs(os.path.dirname(ogb_model_save_path), exist_ok=True)
            # ----------------------------
            best_epoch = 0
            t = time.time()

            for epoch in range(1, self.args.max_epochs + 1):
                train_loss, _, _, _ = self.run_epoch(train_loader)

                def eval_ogb(loader):
                    self.model.eval()
                    y_true, y_pred = [], []
                    for data in loader:
                        data = data.to(self.device)
                        with torch.no_grad():
                            out, _ = self._unpack_output(self.model(data))
                        out = torch.sigmoid(out)
                        if out.shape[-1] == 2 and data.y.shape[-1] == 1:
                            out_to_eval = out[:, 1:2]
                        else:
                            out_to_eval = out

                        y_true.append(data.y.view(out_to_eval.shape).cpu())
                        y_pred.append(out_to_eval.cpu())

                    y_true = torch.cat(y_true, dim=0).numpy()
                    y_pred = torch.cat(y_pred, dim=0).numpy()
                    input_dict = {"y_true": y_true, "y_pred": y_pred}
                    return self.evaluator.eval(input_dict)[self.evaluator.eval_metric]

                val_perf = eval_ogb(val_loader)

                is_better = (val_perf > best_val_perf if 'classification' in self.data.task_type else val_perf < best_val_perf)

                if is_better:
                    best_val_perf = val_perf
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_model(ogb_model_save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}! No improvement in {patience} epochs.")
                        break

            # if best_model_state is not None:
            #     self.model.load_state_dict(best_model_state)
            #     print("Restored the best model weights based on validation performance.")

            if os.path.exists(ogb_model_save_path):
                self.load_model(ogb_model_save_path)
                best_val_perf = eval_ogb(test_loader)
                print(
                    f"Restored best model from '{ogb_model_save_path}' (Best Val Perf: {best_val_perf:.4f} at Epoch {best_epoch}).")
            # ----------------------------

            print(f"OGB Optimization Finished! Best Val: {best_val_perf:.4f}, Test Perf: {best_test_perf:.4f}")
            return best_val_perf, 0, best_test_perf, 0, time.time() - t, 0

        # iterate over 10 folds
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*self.k_fold())):

            # Reinitialise model and optimizer for each fold
            self.model = self.add_model()
            self.optimizer = self.add_optimizer()
            self.lr_scheduler = self.add_lr_scheduler()

            self.model_model = torch.compile(self.model)

            train_dataset = self.data[train_idx]
            test_dataset = self.data[test_idx]
            val_dataset = self.data[val_idx]

            if 'adj' in train_dataset[0]:
                train_loader = DenseDataLoader(train_dataset, self.args.batch_size, shuffle=True)
                val_loader = DenseDataLoader(val_dataset, self.args.batch_size, shuffle=False)
                test_loader = DenseDataLoader(test_dataset, self.args.batch_size, shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, self.args.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, self.args.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, self.args.batch_size, shuffle=False)

            loss_save_path = '{}/{}/seed_{}_fold_{}_loss_and_acc.csv'.format(self.args.directory_path, self.args.log_db,
                                                                             self.args.counter + 1, fold + 1)
            loss_save_list = []

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            best_val_acc, best_test_acc = 0.0, 0.0
            best_val_loss = float('inf')
            best_epoch = 0
            patience_cnt = 0
            t = time.time()
            for epoch in range(1, self.args.max_epochs + 1):
                train_loss, cls_loss, extra_loss, train_acc = self.run_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)

                loss_save_list.append({"Epoch": epoch, "loss_train": train_loss, "acc_train": train_acc,
                                       "loss_val": val_loss, "acc_val": val_acc})
                loss_save = pd.DataFrame(loss_save_list)
                os.makedirs(os.path.dirname(loss_save_path), exist_ok=True)

                loss_save.to_csv(loss_save_path, index=False)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # best_val_loss = val_loss
                    self.save_model(best_val_model_save_path)
                    best_epoch = epoch
                    patience_cnt = 0
                elif val_acc == best_val_acc and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(best_val_model_save_path)
                    best_epoch = epoch
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == self.args.patience:
                    break

            train_times.append(time.time() - t)
            print('Fold {} Optimization Finished! Total time elapsed: {:.6f} s Best epoch: {:02d}'.format(fold + 1,
                                                                                                          time.time() - t,
                                                                                                          best_epoch))
            # load best model for testing
            self.load_model(best_val_model_save_path)
            best_test_loss, best_test_acc, cm, cr = self.predict(test_loader)

            if self.task_type == 'classification':
                print('Seed: {:02d}/{:02d}'.format(self.args.counter + 1, self.args.replication_num),
                      'Fold: {:02d}/{:02d}'.format(fold + 1, self.args.folds),
                      'loss_test: {:.6f}'.format(best_test_loss), 'acc_test: {:.6f}'.format(best_test_acc)
                      )
                print("Confusion Matrix: \n", cm)
                print("Confusion Report: \n", cr)
            else:
                print('Seed: {:02d}/{:02d}'.format(self.args.counter + 1, self.args.replication_num),
                      'Fold: {:02d}/{:02d}'.format(fold + 1, self.args.folds),
                      'loss_test: {:.6f}'.format(best_test_loss), 'mse_test: {:.6f}'.format(best_test_acc)
                      )
                print("Mean Squared Error, Root Mean Squared Error: \n", best_test_acc, np.sqrt(best_test_acc))
                print("Mean Absolute Error: \n", cm)
                print("R²: \n", cr)

            run_test_acc_list.append({"Fold": fold + 1, 'loss_test': best_test_loss, 'acc_test': best_test_acc})
            run_test_acc_save = pd.DataFrame(run_test_acc_list)
            run_test_acc_save.to_csv(run_test_acc_path, index=False)

            with open(run_test_result_path, mode="a") as f:
                f.write(
                    'Fold {} Optimization Finished! Total time elapsed: {:.6f} s Best epoch: {:02d} \n'.format(fold + 1,
                                                                                                               time.time() - t,
                                                                                                               best_epoch))
                f.write('loss_test: {:.6f},  acc_test: {:.6f} \n'.format(best_test_loss, best_test_acc))
                f.write("Confusion Matrix: \n")
                print(cm, file=f)
                f.write("Confusion Report: \n")
                print(cr, file=f)

            if best_test_acc > best_fold_test_acc:
                best_fold_test_acc = best_test_acc
                self.save_model(best_test_model_save_path)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            val_accs.append(best_val_acc)
            test_accs.append(best_test_acc)

        train_time_mean = np.round(np.mean(train_times), 6)
        val_acc_mean = np.round(np.mean(val_accs), 6)
        test_acc_mean = np.round(np.mean(test_accs), 6)

        train_time_std = np.round(np.std(train_times), 6)
        val_acc_std = np.round(np.std(val_accs), 6)
        test_acc_std = np.round(np.std(test_accs), 6)

        return val_acc_mean, val_acc_std, test_acc_mean, test_acc_std, train_time_mean, train_time_std


def train():
    if not args.restore:
        args.name = time.strftime('%Y_%m_%d') + '_' + time.strftime('%H_%M_%S') + '_' + args.name

    # Model training
    print('Training Start ...')

    if args.replication:
        seeds = [3653]
    else:
        seeds = [args.seed]
    args.replication_num = len(seeds)

    counter = 0
    args.log_db = args.name
    print("log_db:", args.log_db)
    args.directory_path = 'results/{}/{}/layer{}_batch{}_hid{}_epoch{}_early{}_pooling{}_drop{}_lr{}_decay{}_JK_{' \
                          '}_others{}/'.format(
        args.model, args.dataset, args.num_layers, args.batch_size, args.hid_dim, args.max_epochs, args.patience,
        args.pooling_ratio, args.dropout_ratio, args.lr, args.weight_decay, args.jump_connection, args.notes)
    print(args)
    make_directory(args.directory_path)
    test_result_path = '{}/{}_all_results.txt'.format(args.directory_path, args.log_db)
    test_acc_path = '{}/{}_all_loss_and_acc.csv'.format(args.directory_path, args.log_db)
    test_acc_list = []

    avg_val = []
    avg_test = []
    avg_time = []
    for seed in seeds:
        # Set seed
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        set_gpu(args.gpu)
        args.counter = counter
        args.name = '{}_seed_{}'.format(args.log_db, counter)

        with open(test_result_path, mode="a") as f:
            print(args, file=f)

        # start training the model
        model = Trainer(args)
        val_acc, val_acc_std, test_acc, test_acc_std, train_time, train_time_std = model.run()
        print('For seed {}  Val Accuracy Mean: {:.6f} ± {:.6f}  Test Accuracy Mean: {:.6f} ± {:.6f}  Train Time Mean: '
              '{:.6f} ± {:.6f} \n'.format(seed, val_acc,
                                          val_acc_std,
                                          test_acc,
                                          test_acc_std,
                                          train_time,
                                          train_time_std))
        with open(test_result_path, mode="a") as f:
            f.write(
                'For seed {} ({:02d}/{:02d})  Val Accuracy: {:.6f}   Test Accuracy: {:.6f}   Train Time: {:.6f} s\n'.format(
                    seed, counter + 1, args.replication_num, val_acc, test_acc, train_time))

        if counter == 0:
            with open(test_result_path, mode="a") as f:
                print(torchinfo.summary(model=model.model), file=f)

        test_acc_list.append(
            {"Seed No.": counter + 1, "Seed": seed, "Val Accuracy": val_acc, "Test Accuracy": test_acc})
        test_acc_save = pd.DataFrame(test_acc_list)
        test_acc_save.to_csv(test_acc_path, index=False)

        avg_val.append(val_acc)
        avg_test.append(test_acc)
        avg_time.append(train_time)
        counter += 1

    print('Val Accuracy: {:.4f} ± {:.4f} Test Accuracy: {:.4f} ± {:.4f} Train Time: {:.3f} ± {:.3f} s'.format(
        np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test), np.mean(avg_time), np.std(avg_time)))
    with open(test_result_path, mode="a") as f:
        f.write('Val Accuracy: {:.4f} ± {:.4f} Test Accuracy: {:.4f} ± {:.4f} Train Time: {:.4f} ± {:.4f} s'.format(
            np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test), np.mean(avg_time),
            np.std(avg_time)))
    return args.log_db, np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test), np.mean(
        avg_time), np.std(avg_time)


if __name__ == '__main__':
    train()