import torch
import torch.nn as nn
import warnings
from monai.utils import pytorch_after
class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes

# giveup
class BoundaryDoULoss3D_V2(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss3D_V2, self).__init__()
        self.n_classes = n_classes
        self.alpha = nn.Parameter(torch.tensor(0.8)).to(device="cuda:0" if torch.cuda.is_available() else "cpu")  # 初始化 alpha 为可学习参数

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):

        alpha = self.alpha  # 使用可学习的 alpha
        alpha = alpha.clamp(min=0.0, max=1.0)  # 限制 alpha 的最大值为 0.8


        smooth = 1e-5

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)
        print("debug:", alpha.item())
        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    
class BoundaryDoULoss3D(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss3D, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        padding_out = torch.zeros((target.shape[0], target.shape[-3] + 2, target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1, 1:-1] = target
        d, h, w = 3, 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - d + 1, padding_out.shape[2] - h + 1,
                        padding_out.shape[3] - w + 1)).cuda(target.device.index)
        for i in range(Y.shape[0]):
            Y[i, :, :,:] = torch.conv3d(target[i].unsqueeze(0),kernel.unsqueeze(0).cuda(target.device.index), padding=1)
        Y = Y * target
        Y[Y == 7] = 0
        C = torch.count_nonzero(Y)  # 表面积
        S = torch.count_nonzero(target)  # 体积
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    
class BoundaryDoUCELoss3D(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoUCELoss3D, self).__init__()
        self.n_classes = n_classes
        self.cross_entropy = nn.CrossEntropyLoss(weight=None, reduction="mean")
        self.old_pt_ver = not pytorch_after(1, 10)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        padding_out = torch.zeros((target.shape[0], target.shape[-3] + 2, target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1, 1:-1] = target
        d, h, w = 3, 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - d + 1, padding_out.shape[2] - h + 1,
                        padding_out.shape[3] - w + 1)).cuda(target.device.index)
        for i in range(Y.shape[0]):
            Y[i, :, :,:] = torch.conv3d(target[i].unsqueeze(0),kernel.unsqueeze(0).cuda(target.device.index), padding=1)
        Y = Y * target
        Y[Y == 7] = 0
        C = torch.count_nonzero(Y)  # 表面积
        S = torch.count_nonzero(target)  # 体积
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])

        bdou_loss = loss / self.n_classes
        ce_loss = self.ce(inputs, target)
        return bdou_loss + ce_loss
    
if __name__=='__main__':
    output = torch.randn(2, 14, 96, 96, 96).to(device='cuda')
    label = torch.randn(2, 1, 96, 96, 96).to(device='cuda')

    loss_function = BoundaryDoULoss3D(
        n_classes=14
    )

    loss = loss_function(output, label)

    print(loss)