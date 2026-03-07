import time
from pathlib import Path
import torch
from argoverse.evaluation.competition_util import generate_forecasting_h5
from torch import Tensor


class SubmissionAv1:
    def __init__(self, save_dir: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.output_path = Path(save_dir)
        self.filename = 'submission_{}'.format(stamp)
        self.test_traj_output = dict()
        self.test_prob_output = dict()


    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
        inference=False,
    ) -> None:
        """
        trajectory: (B, M, 30, 2)
        probability: (B, M)
        normalized_probability: if the input probability is normalized,
        """

        scenario_ids = data["scenario_id"]
        track_ids = data["agent_idx"]
        batch = len(track_ids)

        origin = data["origin"].view(batch, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).reshape(batch, 2, 2)

        with torch.no_grad():
            global_trajectory = (
                torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu().numpy() # BxMxTx2
        probability = probability.detach().cpu().numpy() # BxM

        if inference:
            return global_trajectory, probability

        for i, scenario_id in enumerate(scenario_ids):
            self.test_traj_output[int(scenario_id)] = global_trajectory[i]
            self.test_prob_output[int(scenario_id)] = probability[i]


    def generate_submission_file(self):
        print("generating submission file for argoverse 1 motion forecasting challenge")
        generate_forecasting_h5(self.test_traj_output, self.output_path, self.filename, self.test_prob_output)
        print(f"file saved to {self.filename}")
