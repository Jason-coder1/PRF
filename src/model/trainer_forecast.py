import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av1 import SubmissionAv1
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from omegaconf import OmegaConf

#from .model_forecast_av2_direct import ModelForecast
from .model_forecast_av2 import ModelForecast
#from .model_forecast_av1 import ModelForecast
torch.cuda.empty_cache()


class Trainer(pl.LightningModule):
    def __init__(
        self,
        alignment_pairs,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        align_weight_start: float = 0.05, 
        align_weight_end: float = 0.5,   
        align_weight_warmup_epochs: int = 10,
    ) -> None:
        super(Trainer, self).__init__()

        alignment_pairs = OmegaConf.to_container(alignment_pairs, resolve=True)
        new_alignment_pairs = {}
        for pair in alignment_pairs:
            new_alignment_pairs[tuple(pair[0])] = tuple(pair[1])
        self.alignment_pairs = new_alignment_pairs

        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.align_weight_start = align_weight_start
        self.align_weight_end = align_weight_end
        self.align_weight_warmup_epochs = align_weight_warmup_epochs

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        self.val_metrics_final = metrics.clone(prefix="val_final_")

        self.net = ModelForecast(**model)

        # --- for validation-time benchmarking ---
        self._val_total_time_s = 0.0
        self._val_total_batches = 0
        self._val_total_samples = 0
        self._val_flops_per_sample = None  # lazy-compute on first batch


    def get_current_align_weight(self):
        current_epoch = self.current_epoch
        if current_epoch < self.align_weight_warmup_epochs:
            progress = current_epoch / self.align_weight_warmup_epochs
            weight = self.align_weight_start + \
                    0.5 * (self.align_weight_end - self.align_weight_start) * \
                    (1 - math.cos(progress * math.pi))
        else:
            weight = self.align_weight_end
        return weight

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            prediction, prob = self.submission_handler.format_data(
                cur_data[0], out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data_list, tag=''):
        assert isinstance(data_list, list), "Input data should be a list of dictionaries."

        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        in_batch = len(data_list)
        hist_valid_mask = data_list[0]["x_valid_mask"] # shape: [B, N, T_o] N表示 agent 个数，T_o 表示历史轨迹长度, False is not exists for ego-agent
        hist_predict = out.get("hist_predict", None)
        if hist_predict is not None:
            hist_predict = torch.chunk(hist_predict, chunks=(in_batch-1), dim=0)
            hist_predict = torch.concat(hist_predict, dim=1)  # [B, T, 2]
            imp_length = hist_predict.shape[1]
            x = data_list[0]["x_positions"][:, 0, :imp_length] # [B, T, 2] T = T_o - 10

        hist_pred_others = out.get("hist_others", None)
        if hist_pred_others is not None:
            hist_pred_others = torch.chunk(hist_pred_others, chunks=(in_batch-1), dim=0)
            hist_pred_others = torch.concat(hist_pred_others, dim=2) # [B, N-1, T, 2]
            imp_length = hist_pred_others.shape[2]
            valid_mask_others = hist_valid_mask[:, 1:, :imp_length] # shape: [B, N, T]
            x_others = data_list[0]["x_positions"][:, 1:, :imp_length] # 

        goal_pred = out.get("goal_predict", None)
        if goal_pred is not None:
            #goal_pred = torch.chunk(goal_pred, chunks=(in_batch-1), dim=0) # B, M, 2
            #goal_pred = torch.concat(goal_pred, dim=-1)  # [b, M, 8]
            #x_goals = data_list[0]["x_positions"][:, 0, ::10][:, :-1].reshape(-1, goal_pred.shape[-1]) # [b, 5, 2] -> [b, 4, 2] -> [B, 2]
            '''
            x_goals = data_list[0]["x_positions"][:, 0, ::10][:, :-1]  # [b, 4, 2]
            x_goals = torch.chunk(x_goals, chunks=(in_batch-1), dim=1)  # [b, 1, 2] x 4
            x_goals = torch.concat(x_goals, dim=0).squeeze(1)  # [B, 1, 2]
            '''
            x_goals = data_list[0]["x_positions"][:, 0, :imp_length]  # [b, 40, 2]
            x_goals = torch.chunk(x_goals, chunks=(in_batch-1), dim=1)  # [b, 10, 2] x 4
            x_goals = torch.concat(x_goals, dim=0)  # [B, 10, 2]
            #B, _, _ = x_goals.shape
            #x_goals_flat = x_goals.reshape(B, -1)  # [B, 8]

        hist_x_hat = out.get("hist_x_hat", None)
        hist_pi = out.get("hist_pi", None)
        hist_scal = out.get("hist_scal", None)

        if hist_x_hat is not None and hist_pi is not None and hist_scal is not None:
            x_sb = data_list[0]["x_positions"][:, 0, :imp_length] # [B, T, 2] T = T_o - 10
            x_sb = torch.chunk(x_sb, chunks=(in_batch-1), dim=1)  # [B, T/4, 2]
            x_sb = torch.concat(x_sb, dim=0)  # [4B, T/4, 2]

        # gt
        y_list, y_others_list = [], []
        for data in data_list:
            y, y_others = data["target"][:, 0], data["target"][:, 1:]
            y_list.append(y)
            y_others_list.append(y_others)
        
        y = torch.cat(y_list, dim=0)
        y_others = torch.cat(y_others_list, dim=0)

        # loss for output of history query
        if hist_predict is not None:
            hist_reg_loss = F.smooth_l1_loss(hist_predict, x)
        else:
            hist_reg_loss = 0

        # loss for output of state query
        if dense_predict is not None:
            dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0

        # loss for output of goal
        if goal_pred is not None:
            goal_pred = goal_pred.view(goal_pred.size(0), goal_pred.size(1), -1, 2)  # [B, M, 10, 2]
            l2_norm = torch.norm(goal_pred - x_goals.unsqueeze(1), dim=-1).sum(dim=-1) # sum(dim=-1)的作用 [B, M, 10] -> [B, M]
            best_goal = torch.argmin(l2_norm, dim=-1)
            goal_pred_best = goal_pred[torch.arange(goal_pred.shape[0]), best_goal]
            goal_reg_loss = F.smooth_l1_loss(goal_pred_best, x_goals)
        else:
            goal_reg_loss = 0

        # loss for output of mode query
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        
        # loss for final output
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for hist output
        if hist_x_hat is not None:
            l2_norm_hist = torch.norm(hist_x_hat[..., :2] - x_sb.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_hist = torch.argmin(l2_norm_hist, dim=-1)
            hist_x_hat_best = hist_x_hat[torch.arange(hist_x_hat.shape[0]), best_mode_hist]
            hist_agent_reg_loss = F.smooth_l1_loss(hist_x_hat_best[..., :2], x_sb)
        else:
            hist_agent_reg_loss = 0
        if hist_pi is not None:
            hist_pi_reg_loss = F.cross_entropy(hist_pi, best_mode_hist.detach(), label_smoothing=0.2)
        else:
            hist_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask_list = []
        for data in data_list:
            others_reg_mask = data["target_mask"][:, 1:]
            others_reg_mask_list.append(others_reg_mask)
        others_reg_mask = torch.cat(others_reg_mask_list, dim=0)

        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        if hist_pred_others is not None: # valid_mask_others for only calculate loss for ture observations
            hist_others_reg_loss = F.smooth_l1_loss(
                hist_pred_others[valid_mask_others], x_others[valid_mask_others]
            )
        else:
            hist_others_reg_loss = 0

        ###### decoder stage regression ######

        # final loss using retrospective query
        scal_f = out.get("final_scal", None)
        y_hat_f = out.get("final_y_hat", None)
        pi_f = out.get("final_pi", None)
        ''' # generate offset to refine the final prediction results, a modification of step 3.
        if y_hat_f is not None:
            agent_reg_loss_f = F.smooth_l1_loss(new_y_hat[:y_hat_f.shape[0]]+y_hat_f, y[:y_hat_f.shape[0]].unsqueeze(1))
        '''
        # loss for final output
        if y_hat_f is not None:
            l2_norm_f = torch.norm(y_hat_f[..., :2] - y[:y_hat_f.shape[0]].unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_f = torch.argmin(l2_norm_f, dim=-1)
            y_hat_best_f = y_hat_f[torch.arange(y_hat_f.shape[0]), best_mode_f]
            agent_reg_loss_f = F.smooth_l1_loss(y_hat_best_f[..., :2], y[:y_hat_f.shape[0]])
        else:
            agent_reg_loss_f = 0
        if pi_f is not None:
            pi_reg_loss_f = F.cross_entropy(pi_f, best_mode_f.detach(), label_smoothing=0.2)
        else:
            pi_reg_loss_f = 0
        # Laplace loss
        if y_hat_f is not None and pi_f is not None and scal_f is not None:
            predictions_f = {}
            predictions_f['traj'] = y_hat_f
            predictions_f['scale'] = scal_f
            predictions_f['probs'] = pi_f
            laplace_loss_f = self.laplace_loss.compute(predictions_f, y[:y_hat_f.shape[0]])
        else:
            laplace_loss_f = 0

        # Laplace loss, which is not necessary
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        if hist_x_hat is not None and hist_pi is not None and hist_scal is not None:
            hist_preds = {}
            hist_preds['traj'] = hist_x_hat
            hist_preds['scale'] = hist_scal
            hist_preds['probs'] = hist_pi
            laplace_loss_hist = self.laplace_loss.compute(hist_preds, x_sb)
        else:
            laplace_loss_hist = 0

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + \
                new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss + \
                goal_reg_loss + hist_others_reg_loss + hist_reg_loss + \
                hist_agent_reg_loss + hist_pi_reg_loss + agent_reg_loss_f #+ \
                #pi_reg_loss_f

        loss = loss + laplace_loss + laplace_loss_new + laplace_loss_hist #+ laplace_loss_f

        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
        }
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()

        if hist_predict is not None:
            disp_dict[f"{tag}reg_loss_hist"] = hist_reg_loss.item()
        if hist_pred_others is not None:
            disp_dict[f"{tag}reg_loss_hist_others"] = hist_others_reg_loss.item()
        if goal_pred is not None:
            disp_dict[f"{tag}reg_loss_goal"] = goal_reg_loss.item()
        
        if hist_scal is not None:
            disp_dict[f"{tag}laplace_loss_hist"] = laplace_loss_hist.item()
        if hist_x_hat is not None:
            disp_dict[f"{tag}reg_loss_hist_refine"] = hist_agent_reg_loss.item()
        if hist_pi is not None:
            disp_dict[f"{tag}reg_loss_hist_pi"] = hist_pi_reg_loss.item()

        if scal_f is not None:
            disp_dict[f"{tag}laplace_loss_final"] = laplace_loss_f.item()
        if y_hat_f is not None:
            disp_dict[f"{tag}reg_loss_final"] = agent_reg_loss_f.item()
        if pi_f is not None:
            disp_dict[f"{tag}reg_loss_pi_final"] = pi_reg_loss_f.item()

        return loss, disp_dict

    def group_and_sort_by_hist_end(self, data_list):
        # 先按 hist_end 分组
        hist_end_groups = defaultdict(list)
        for data in data_list:
            hist_end = data.get('hist_end')[0].item()
            hist_end_groups[hist_end].append(data)

        # 把 hist_end 排序（降序）
        sorted_hist_ends = sorted(hist_end_groups.keys(), reverse=True)

        # 按顺序生成 list of lists
        grouped_list = [hist_end_groups[hist_end] for hist_end in sorted_hist_ends]

        return grouped_list

    def training_step(self, data_list, batch_idx):
        total_loss = 0
        align_loss_total = 0
        total_loss_dict = defaultdict(float)

        self.net.teacher_feats.clear() # 对应模型保存的特征字典
        self.net.student_feats.clear()

        # parallel training on a batch with the same hist_end parameter
        #out = self(data_list)
        #loss, loss_dict = self.cal_loss(out, data_list)
        #total_loss += loss

        data_groups = {}
        for data in data_list:
            hist_end = data.get('hist_end')[0].item() 
            if hist_end not in data_groups:
                data_groups[hist_end] = []
            data_groups[hist_end].append(data)

        # 训练不同结束点下的数据batch，比如[0,50], [0,40], [0, 30] -> [50,110], [40,100], [30,90]
        model_loss = 0
        num_data = len(data_list)
        for data_id in sorted(data_groups.keys(), reverse=True):
            data_sublist = data_groups[data_id]
            list_len = len(data_sublist)
            batch_weight = list_len / num_data
            out = self(data_sublist)
            loss, loss_dict = self.cal_loss(out,data_sublist)
            model_loss += loss * batch_weight

            for k, v in loss_dict.items():
                total_loss_dict[k] += v * batch_weight

        # ---- 及时释放 ----
        #del out, loss, loss_dict, data_sublist
        #torch.cuda.empty_cache()   # 清理缓存

        total_loss += model_loss

        # 特征损失
        align_count = 0
        align_loss = 0
        for (student_start, student_end), (teacher_start, teacher_end) in self.alignment_pairs.items():
            student_feat = self.net.student_feats.get((student_start, student_end))
            teacher_feat = self.net.teacher_feats.get((teacher_start, teacher_end))
            if student_feat is not None and teacher_feat is not None:
                align_weight = self.get_current_align_weight()
                with torch.no_grad():
                    teacher_feat_detached = teacher_feat.detach()
                align_loss = nn.L1Loss()(student_feat, teacher_feat_detached)
                align_loss_total += align_loss * align_weight
                align_count += 1

        if align_count > 0:
            align_loss_total /= align_count
        
        total_loss += align_loss_total

        for k, v in total_loss_dict.items(): # loss_dict -> total_loss_dict
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return total_loss

    '''
    def on_validation_epoch_start(self):
        # reset accumulators
        self._val_total_time_s = 0.0
        self._val_total_batches = 0
        self._val_total_samples = 0
        self._val_flops_per_sample = None

    def on_validation_epoch_end(self):
        # aggregate & log epoch-level averages
        if self._val_total_batches == 0:
            return
        avg_ms_per_batch = (self._val_total_time_s / self._val_total_batches) * 1000.0
        avg_ms_per_sample = (self._val_total_time_s / max(1, self._val_total_samples)) * 1000.0
        throughput = (self._val_total_samples / self._val_total_time_s) if self._val_total_time_s > 0 else float("inf")

        # 这些是“验证集平均”指标
        self.log("val/avg_latency_ms_per_batch", avg_ms_per_batch, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/avg_latency_ms_per_sample", avg_ms_per_sample, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/throughput_samples_per_s", throughput, prog_bar=True, on_epoch=True, sync_dist=True)

        # FLOPs（按样本）；batch 的 FLOPs = per_sample * 平均 batch_size
        if self._val_flops_per_sample is not None and not (isinstance(self._val_flops_per_sample, float) and math.isnan(self._val_flops_per_sample)):
            self.log("val/FLOPs_per_sample", self._val_flops_per_sample, on_epoch=True, sync_dist=True)
            avg_bs = self._val_total_samples / self._val_total_batches
            self.log("val/FLOPs_per_batch", self._val_flops_per_sample * avg_bs, on_epoch=True, sync_dist=True)
    '''

    def validation_step(self, data, batch_idx):
        '''
        # ===== 计时（只包住前向，不含指标计算）=====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            _start = torch.cuda.Event(enable_timing=True)
            _end = torch.cuda.Event(enable_timing=True)
            _start.record()
            out = self(data)   # ← 你的前向
            _end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(_start.elapsed_time(_end))
        else:
            t0 = time.perf_counter()
            out = self(data)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 推断 batch size（按你的数据结构）
        try:
            bs = int(data[0]['target'].shape[0])
        except Exception:
            bs = None

        # 累计
        self._val_total_time_s += elapsed_ms / 1000.0
        self._val_total_batches += 1
        if bs is not None:
            self._val_total_samples += bs

        # ====== FLOPs（仅第一个可用 batch；会额外跑一次前向用于分析）======
        if self._val_flops_per_sample is None and bs is not None:
            try:
                from fvcore.nn import FlopCountAnalysis
                # 注：这里直接对 self.net 做分析，输入与 forward 一致
                with torch.inference_mode():
                    flops_batch = FlopCountAnalysis(self.net, (data,)).total()  # 次/批
                self._val_flops_per_sample = flops_batch / bs
            except Exception as e1:
                try:
                    from thop import profile
                    with torch.inference_mode():
                        macs, _ = profile(self.net, inputs=(data,), verbose=False)
                    self._val_flops_per_sample = (macs * 2.0) / bs  # MACs≈FLOPs/2
                except Exception as e2:
                    self._val_flops_per_sample = float("nan")
                    self.print(f"[FLOPs 统计失败] fvcore: {e1}; thop: {e2}")
        '''

        #if isinstance(data, list):
        #    data = data[0]
        out = self(data)
        #_, loss_dict = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data[0]['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            #if out['final_y_hat'] is not None:
            #    out['y_hat'] += out['final_y_hat']
            out['pi'] = out['new_pi']
            metrics_new = self.val_metrics_new(out, data[0]['target'][:, 0])

        #if out['final_y_hat'] is not None:
        #    out['y_hat'] = out['final_y_hat']
        #    out['pi'] = out['final_pi']
        #    metrics_final = self.val_metrics_final(out, data[0]['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
        #if out['final_y_hat'] is not None:
        #    self.log_dict(
        #        metrics_final,
        #        prog_bar=True,
        #        on_step=False,
        #        on_epoch=True,
        #        batch_size=1,
        #        sync_dist=True,
        #    )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2( # SubmissionAv1 for evaluating argoverse 1 dataset
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        #if isinstance(data, list):
        #    data = data[0]
        out = self(data)
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data[0], out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,)

        return [optimizer], [scheduler]