import pretrainedmodels
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Sequential
from torch import nn
import math
import torch
from typing import List


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LazyLoadModule(nn.Module):
    """Lazy buffer/parameter loading using load_state_dict_pre_hook

    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and
    save buffer with `register_buffer`/`register_parameter`
    method, which can be outside of __init__ method.
    Then this module can load any shape of Tensor during de-serializing.

    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.
    """
    _lazy_buffer_keys: List[str] = []     # It needs to be override to register lazy buffer
    _lazy_parameter_keys: List[str] = []  # It needs to be override to register lazy parameter

    def __init__(self):
        super(LazyLoadModule, self).__init__()
        for k in self._lazy_buffer_keys:
            self.register_buffer(k, torch.tensor([]))
        for k in self._lazy_parameter_keys:
            self.register_parameter(k, None)
        self._register_load_state_dict_pre_hook(self._hook)

    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])

        for key in self._lazy_parameter_keys:
            self.register_parameter(key, Parameter(state_dict[prefix + key]))


class LazyLinear(LazyLoadModule):
    """Linear module with lazy input inference

    `in_features` can be `None`, and it is determined at the first time of forward step dynamically.
    """

    __constants__ = ['bias', 'in_features', 'out_features']
    _lazy_parameter_keys = ['weight']

    def __init__(self, in_features, out_features, bias=True):
        super(LazyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if in_features is not None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is None:
            self.in_features = input.shape[-1]
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.reset_parameters()

            # Need to send lazy defined parameter to device...
            self.to(input.device)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        if in_features is None:
            self.linear = LazyLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h

class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=1, out_dim=10, use_bn=True,
                 pretrained=None):
        super(PretrainedCNN, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        activation = F.leaky_relu
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h

class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self, x, y=None, no_jsd = False):
        if no_jsd:
            pred = self.predictor(x)
            if isinstance(pred, tuple):
                assert len(pred) == 3
                preds = pred
            else:
                assert pred.shape[1] == self.n_total_class
                preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
            
            loss_grapheme = F.cross_entropy(preds[0], y[:,0])
            loss_vowel = F.cross_entropy(preds[1], y[:,1])
            loss_consonant = F.cross_entropy(preds[2], y[:,2])

            acc1_grapheme, acc5_grapheme = accuracy(preds[0], y[:,0], topk=(1, 5))
            acc1_vowel, acc5_vowel = accuracy(preds[1], y[:,1], topk=(1, 5))
            acc1_consonant, acc5_consonant = accuracy(preds[2], y[:,2], topk=(1, 5))
        else:
            x_i = torch.cat(x, 0).to(device)
            logits_all = self.predictor(x_i)
            pred, preds_aug1, preds_aug2 = torch.split(logits_all, x[0].size(0))
            if isinstance(pred, tuple):
                assert len(pred) == 3
                preds = pred 
            else:
                assert pred.shape[1] == self.n_total_class
                preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
                preds_aug1 = torch.split(preds_aug1, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
                preds_aug2 = torch.split(preds_aug2, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
            
            loss_grapheme = F.cross_entropy(preds[0], y[:,0])
            loss_vowel = F.cross_entropy(preds[1], y[:,1])
            loss_consonant = F.cross_entropy(preds[2], y[:,2])

            p_clean_grapheme, p_aug1_grapheme, p_aug2_grapheme = F.softmax(preds[0], dim=1), F.softmax(preds_aug1[0], dim=1), F.softmax(preds_aug2[0], dim=1)
            p_clean_vowel, p_aug1_vowel, p_aug2_vowel = F.softmax(preds[1], dim=1), F.softmax(preds_aug1[1], dim=1), F.softmax(preds_aug2[1], dim=1)
            p_clean_consonant, p_aug1_consonant, p_aug2_consonant = F.softmax(preds[2], dim=1), F.softmax(preds_aug1[2], dim=1), F.softmax(preds_aug2[2], dim=1)

            p_mixture_grapheme = torch.clamp((p_clean_grapheme + p_aug1_grapheme + p_aug2_grapheme) / 3., 1e-7, 1).log()
            p_mixture_vowel = torch.clamp((p_clean_vowel + p_aug1_vowel + p_aug2_vowel) / 3., 1e-7, 1).log()
            p_mixture_consonant = torch.clamp((p_clean_consonant + p_aug1_consonant + p_aug2_consonant) / 3., 1e-7, 1).log()

            loss_grapheme += 12 * (F.kl_div(p_mixture_grapheme, p_clean_grapheme, reduction='batchmean') +
                            F.kl_div(p_mixture_grapheme, p_aug1_grapheme, reduction='batchmean') +
                            F.kl_div(p_mixture_grapheme, p_aug2_grapheme, reduction='batchmean')) / 3.
            loss_vowel += 12 * (F.kl_div(p_mixture_vowel, p_clean_vowel, reduction='batchmean') +
                            F.kl_div(p_mixture_vowel, p_aug1_vowel, reduction='batchmean') +
                            F.kl_div(p_mixture_vowel, p_aug2_vowel, reduction='batchmean')) / 3.
            loss_consonant += 12 * (F.kl_div(p_mixture_consonant, p_clean_consonant, reduction='batchmean') +
                            F.kl_div(p_mixture_consonant, p_aug1_consonant, reduction='batchmean') +
                            F.kl_div(p_mixture_consonant, p_aug2_consonant, reduction='batchmean')) / 3.
            acc1_grapheme, acc5_grapheme = accuracy(preds[0], y[:,0], topk=(1, 5))
            acc1_vowel, acc5_vowel = accuracy(preds[1], y[:,1], topk=(1, 5))
            acc1_consonant, acc5_consonant = accuracy(preds[2], y[:,2], topk=(1, 5))
        
        loss = loss_grapheme + loss_vowel + loss_consonant
        metrics = {
            'loss': loss.item(),
            'loss_grapheme': loss_grapheme.item(),
            'loss_vowel': loss_vowel.item(),
            'loss_consonant': loss_consonant.item(),
            'acc1_grapheme': acc1_grapheme.item(), 'acc5_grapheme': acc5_grapheme.item(),
            'acc1_vowel': acc1_vowel.item(), 'acc5_vowel':acc5_vowel.item(),
            'acc1_consonant': acc1_consonant.item(), 'acc5_consonant':acc5_consonant.item()
        }
        return loss, metrics

    def calc(self, data_loader):
        device: torch.device = next(self.parameters()).device
        self.eval()
        output_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # TODO: support general preprocessing.
                # If `data` is not `Data` instance, `to` method is not supported!
                batch = batch.to(device)
                pred = self.predictor(batch)
                output_list.append(pred)
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds

    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels

def build_predictor(arch, out_dim, model_name=None):
    if arch == 'pretrained':
        predictor = PretrainedCNN(in_channels=1, out_dim=out_dim, model_name=model_name)
    else:
        raise ValueError("[ERROR] Unexpected value arch={}".format(arch))
    return predictor

def build_classifier(arch, load_model_path, n_total, model_name='se_resnext101_32x4d', device='cuda:0'):
    if isinstance(device, str):
        device = torch.device(device)
    predictor = build_predictor(arch, out_dim=n_total, model_name=model_name)
    print('predictor', type(predictor))
    classifier = BengaliClassifier(predictor)
    if load_model_path:
        predictor.load_state_dict(torch.load(load_model_path))
    else:
        print("[WARNING] Unexpected value load_model_path={}"
              .format(load_model_path))
    classifier.to(device)
    return classifier