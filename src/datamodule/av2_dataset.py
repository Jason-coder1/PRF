from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        candidate_times: List[int] = [0, 10, 20, 30, 40, 50],
        radius: float = 150.0,
        train_mode: str = 'only_focal',
        val_squence_start: int = 0,
    ):
        assert train_mode in ['only_focal', 'focal_and_scored']
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()                      
        self.split = split
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        #if split == 'train':
        #    self.file_list = self.file_list[::2]    # 只使用一半的数据进行训练
        #if split == 'val':
        #    self.file_list = self.file_list[::1]   # 验证集使用八分之一的数据
        self.num_future_steps = 0 if split =='test' else 60
        self.candidate_times = candidate_times
        self.mode = 'only_focal' if split != 'train' else train_mode
        self.radius = radius
        self.val_squence_start = val_squence_start

        print(
            f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}'
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        data = self.process(data)
        return data
    
    def process(self, data):
        sequence_data = []
        train_idx = [data['focal_idx']]
        
        # 'only_focal' for single-agent setting, 'focal_and_scored' for multi-agent setting
        if self.mode == 'focal_and_scored':
            train_idx += data['scored_idx']
        
        if self.split == "val":
            segments = [(self.val_squence_start, 50)]
        elif self.split == "test":
            segments = [(0, 50)]
        else:
            # [0, 10] [0, 20] [0, 30] [0, 40] [0, 50]
            # [10, 20] [10, 30] [10, 40] [10, 50]
            # [20, 30] [20, 40] [20, 50]
            # [30, 40] [30, 50]
            # [40, 50]
            segments = []
            for i in range(len(self.candidate_times)): 
                for j in range(i+1, len(self.candidate_times)):
                    if j == 1: #or j == 2: # j = 1, 2, 3, ... means don't use hist_end=10, 20, 30, ... for training
                        continue
                    segments.append((self.candidate_times[i], self.candidate_times[j]))
            
            #segments = [(0, 50), (10, 50), (20, 50), (30, 50), (40, 50)] # only use hist_end = 50 for training

        for hist_start, hist_end in segments:
            for ag_idx in train_idx:
                ag_dict = self.process_single_agent(data, ag_idx, hist_start, hist_end)
                sequence_data.append(ag_dict)
        
        return sequence_data

    def process_single_agent(self, data, idx, hist_start, hist_end):
        # info for cur_agent on cur_step
        hist_len = hist_end - hist_start  # 动态计算历史轨迹的长度
        # hist_end 作为当前时刻
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, hist_end - 1].double()
        theta = data['x_angles'][idx, hist_end - 1].double()
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, hist_end - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, hist_end - 1]
        ag_mask[idx] = False

        # transform agents to local
        st, ed = hist_start, hist_end + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[idx]], vel[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        pos[valid_mask] = torch.matmul(pos[valid_mask].double() - origin, rotate_mat).to(torch.float32)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2).double() - origin, rotate_mat).reshape(-1, l_pos.size(1), 2).to(torch.float32)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, hist_len - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[0] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        vel = vel[ag_mask]
        attr = attr[ag_mask]
        valid_mask = valid_mask[ag_mask]

        # post_process
        head = head[:, :hist_len]
        vel_future = vel[:, hist_len:]
        vel = vel[:, :hist_len]
        pos_ctr = pos[:, hist_len - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :hist_len], pos[:, hist_len:]
            target_mask = type_mask & valid_mask[:, [hist_len - 1]] & valid_mask[:, hist_len:]
            valid_mask = valid_mask[:, :hist_len]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros(pos_ctr.size(0), 60, 2),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :hist_len - 1] & valid_mask[:, 1: hist_len]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:hist_len] - pos[:, :hist_len - 1]
        
        # add target velocity and acceleration 
        target_diff = None
        if target is not None:
            target_diff_tmp = torch.cat((pos[:, -1].unsqueeze(1), target), dim=1)
            target_diff = target_diff_tmp[:, 1:self.num_future_steps+1] - target_diff_tmp[:, :self.num_future_steps]
            target_diff_tmp = target_diff.clone()
            diff_mask_target_tmp = torch.cat((valid_mask[:,-1].unsqueeze(1), target_mask), dim=1)
            diff_mask_target = diff_mask_target_tmp[:, 1:self.num_future_steps + 1] & diff_mask_target_tmp[:, : self.num_future_steps]
            target_diff[:, :] = torch.where(
                diff_mask_target.unsqueeze(-1),
                target_diff_tmp, torch.zeros(target_diff_tmp.size(0), self.num_future_steps, 2)
            )
        
        pos[:, 1:hist_len] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), hist_len - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:hist_len] - vel[:, :hist_len - 1]
        vel[:, 1:hist_len] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), hist_len - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))
        
        # add target velocity and acceleration
        if target is not None:
            tmpvel_future = vel_future.clone()
            tmpvel_future = torch.cat((tmp_vel[:, -1].unsqueeze(1), tmpvel_future), dim=1)
            vel_diff_future= tmpvel_future[:, 1:self.num_future_steps+1] - tmpvel_future[:, :self.num_future_steps]
            vel_future[:, :] = torch.where(
                diff_mask_target,
                vel_diff_future, torch.zeros(vel_diff_future.size(0), self.num_future_steps)
            )
        
        # padding zeros to the incomplete observation
        if hist_start > 0 or hist_end < 50:
            pos = self.pad_front(pos, hist_start, hist_end)
            tmp_pos = self.pad_front(tmp_pos, hist_start, hist_end)
            head = self.pad_front(head, hist_start, hist_end)
            tmp_vel = self.pad_front(tmp_vel, hist_start, hist_end)
            vel = self.pad_front(vel, hist_start, hist_end)
            valid_mask = self.pad_front(valid_mask, hist_start, hist_end)


        return {
            'target': target, # N, T_f, 2
            'target_diff': target_diff, # N, T_f, 2
            'target_vel_diff': vel_future, # N, T_f
            'target_mask': target_mask, # # N, T_f

            'x_positions_diff': pos, # N, T_o, 2
            'x_positions': tmp_pos, # N, T_o, 2
            'x_attr': attr, # N, C
            'x_centers': pos_ctr, #	N, 2
            'x_angles': head, # N, T_o
            'x_velocity': tmp_vel, # N, T_o
            'x_velocity_diff': vel, # N, T_o
            'x_valid_mask': valid_mask, # N, T_o
            'hist_start': hist_start, # 1
            'hist_end': hist_end, # 1f
            'hist_len': hist_len, # 1

            'lane_positions': l_pos, # L, P, 2 -> 车道数，每个车道的点数，坐标
            'lane_centers': l_ctr, # L, 2
            'lane_angles': l_head, # L
            'lane_attr': l_attr, # L, D -> D表示属性维度
            'lane_valid_mask': l_valid_mask, # L, P
            'is_intersections': l_is_int, # L
            
            'origin': origin.view(1, 2), # 1, 2
            'theta': theta.view(1), # 1
            'scenario_id': data['scenario_id'], # 标量或字符串
            'track_id': cur_agent_id, # 标量或字符串
            'city': data['city'], # 字符串 
            'timestamp': torch.Tensor([hist_end * 0.1]) # 1
        }

    def pad_front(self, in_data, hist_start, hist_end, pad_dim=1, pad_value=0.0):
        """
        通用前向 padding 函数。
        - 对 bool 类型（如 valid_mask）使用 False 填充；
        - 对其他类型使用指定 pad_value。
        """

        pad_len = hist_start
        if pad_len == 0:
            return in_data

        pad_shape = list(in_data.shape)
        pad_shape[pad_dim] = pad_len

        # 针对 bool 类型（valid_mask）使用 False 填充
        if in_data.dtype == torch.bool:
            pad_data = torch.zeros(pad_shape, dtype=torch.bool, device=in_data.device)
        else:
            pad_data = torch.full(pad_shape, pad_value, dtype=in_data.dtype, device=in_data.device)

        assert hist_end == pad_data.shape[pad_dim] + in_data.shape[pad_dim], \
            f"Padding length {pad_len} + tensor length {in_data.shape[pad_dim]} should equal hist_end {hist_end}."

        return torch.cat([pad_data, in_data], dim=pad_dim)
        '''
        out = torch.cat([pad_data, in_data], dim=pad_dim)

        s, e = 30, 35          # 目标区间：[30, 35) -> 30..34
        ref_i = 35             # 参考位置：35

        # edge-padding
        T = out.shape[pad_dim]
        if ref_i < T:
            e = min(e, T)      # 防越界
            if s < e:
                # 取出 pad_dim=35 的“那一帧”（少一维）
                ref = out.select(pad_dim, ref_i)  # shape: out.shape 去掉 pad_dim

                # 构造目标切片 out[..., 30:35, ...]
                idx = [slice(None)] * out.dim()
                idx[pad_dim] = slice(s, e)

                # 将 ref 扩展成与目标切片同形状，然后赋值
                out[tuple(idx)] = ref.unsqueeze(pad_dim).expand_as(out[tuple(idx)])

        # zero-padding
        T = out.shape[pad_dim]
        e = min(e, T)          # 防越界
        if s < e:
            idx = [slice(None)] * out.dim()
            idx[pad_dim] = slice(s, e)
            out[tuple(idx)] = 0  # zero-padding

        return out
        '''

    def pad_to_fixed_length(
        self,
        in_data,
        hist_start,
        hist_end,
        max_len=50,
        pad_dim=1,
        pad_value=0.0
    ):
        """
        通用前后 padding 函数，将输入张量 pad 到 [0, max_len) 区间。

        参数：
        - in_data: 输入张量
        - hist_start: 起始位置 (0-based)
        - hist_end: 结束位置 (exclusive)
        - max_len: 全局最大长度
        - pad_dim: 时间维
        - pad_value: 非 bool 填充值
        """
        assert hist_start >= 0 and hist_end <= max_len, \
            f"hist_start and hist_end must be within [0, {max_len}]"

        pad_front_len = hist_start
        pad_back_len = max_len - hist_end

        parts = []

        # 前 pad
        if pad_front_len > 0:
            front_shape = list(in_data.shape)
            front_shape[pad_dim] = pad_front_len

            if in_data.dtype == torch.bool:
                front_pad = torch.zeros(front_shape, dtype=torch.bool, device=in_data.device)
            else:
                front_pad = torch.full(front_shape, pad_value, dtype=in_data.dtype, device=in_data.device)

            parts.append(front_pad)

        # 中间原始数据
        parts.append(in_data)

        # 后 pad
        if pad_back_len > 0:
            back_shape = list(in_data.shape)
            back_shape[pad_dim] = pad_back_len

            if in_data.dtype == torch.bool:
                back_pad = torch.zeros(back_shape, dtype=torch.bool, device=in_data.device)
            else:
                back_pad = torch.full(back_shape, pad_value, dtype=in_data.dtype, device=in_data.device)

            parts.append(back_pad)

        # 拼接所有部分
        return torch.cat(parts, dim=pad_dim)

def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        for key in [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if 'x_scored' in batch[0]:
            data['x_scored'] = pad_sequence(
                [b['x_scored'] for b in batch], batch_first=True
            )

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_diff'] = pad_sequence([b['target_diff'] for b in batch], batch_first=True)
            data['target_vel_diff'] = pad_sequence([b['target_vel_diff'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]
        data['hist_start'] = torch.tensor([b['hist_start'] for b in batch])
        data['hist_end'] = torch.tensor([b['hist_end'] for b in batch])
        data['hist_len'] = torch.tensor([b['hist_len'] for b in batch])

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
        
    return seq_data
