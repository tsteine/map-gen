from model import Predictions
from dataclasses import dataclass


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float


def compute_loss(self, preds: Predictions, outcomes: Outcomes, config: LossConfig):
    s = data.action.shape[1]
    preds = self.get_preds(raw_preds)

    mask = (data.action[:, :, 0] != self.state_model.num_rooms - 1).to(raw_preds.dtype)
    # TODO: mask out loss values for dummy actions

    all_binary_outputs = torch.cat([data.door_connects, data.missing_connects], dim=1)
    all_binary_outputs = all_binary_outputs.unsqueeze(1).expand(-1, s, -1)

    state_value_raw_logodds = torch.cat([preds.door_connects, preds.missing_connects], dim=2)
    # print("train idx: ", num_binary_outputs + num_save_dist_outputs, "num_binary_outputs =", num_binary_outputs, "num_save_dist_outputs =", num_save_dist_outputs)

    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(state_value_raw_logodds,
                                                                        all_binary_outputs.to(
                                                                            state_value_raw_logodds.dtype),
                                                                        reduction='none')
    binary_loss = torch.mean(binary_loss * mask.unsqueeze(2))
