import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math
from typing import Optional, List
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Linear8bitLt
from torch.distributions.normal import Normal
from collections import Counter

def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    # BNB requires CUDA weights
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    if is_cpu:
        weight = weight.to(torch.device("cuda"))

    cls_name = weight.__class__.__name__
    if cls_name == "Params4bit":
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized.to(device)
        return dequantized

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(
            weight.data, to_order=state.formatB
        )
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    dequantized = bnb.functional.mm_dequant(
        out32, Sout32, SCim, state.SCB, bias=None
    ).t()
    if is_cpu:
        dequantized = dequantized.to(device)
    return dequantized


def dequantize_module_weight(module: torch.nn.Module) -> torch.nn.Parameter:
    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        weight = module.dequantize()
        return weight

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(
            f"Input weight should be of type nn.Parameter, got {type(weight)} instead"
        )

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    if is_cpu:
        # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
        module.weight = module.weight.to(device)
    return weight


class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        # config: LoraConfig,
        expert_num: int = 8,
        router_num: int = 14,
        use_dora_: bool = False,
        # Rank-Stabilized LoRA
        # sets the adapter scaling factor to `alpha/math.sqrt(r)`
        use_rslora_: bool = False,
        # can be original or gaussian
        lora_init_: str = "original",
        lora_r_: int = 8,
        lora_alpha_: int = 16,
        lora_dropout_: float = 0.05,
        target_modules_: Dict[str, bool] = None,
        weight: Tuple[torch.Tensor, torch.Tensor] = (None, None),
        dtype_: torch.dtype = None,
        device: str = None,
    ):
        super().__init__()

        out_dim, in_dim = base_layer.weight.shape

        self.expert_num = expert_num
        self.router_num = router_num
        self.base_layer_ = base_layer
        self.device_ = torch.device(device) if device else base_layer.weight.device
        self.dtype_ = dtype_

        self.initializer_ = lora_init_
        self.r_ = lora_r_
        self.alpha_ = lora_alpha_


        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.task_id = 0

        if use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        self.in_features_ = in_dim
        self.out_features_ = out_dim

        assert lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=lora_dropout_)

        # self.lora_A = nn.Linear(
        #     self.in_features_,
        #     self.r_,
        #     bias=False,
        #     dtype=self.dtype_,
        #     device=self.device_,
        # )
        # self.lora_B = nn.Linear(
        #     self.r_,
        #     self.out_features_,
        #     bias=False,
        #     dtype=self.dtype_,
        #     device=self.device_,
        # )
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.lora_A_list = nn.ModuleList()
        self.lora_B_list = nn.ModuleList()
        self.router_list = nn.ModuleList()
        
        self.top_k = 2
        
        self.apply_moe = True

        if self.apply_moe == True:
            if self.task_id>-1: # router>1
                self.router_list = nn.ParameterList()
                self.w_noise_list = nn.ParameterList()
                for i in range(self.router_num): # Task number
                    self.router_list.append(nn.Parameter(torch.zeros(in_dim, self.expert_num), requires_grad=True))
                    self.w_noise_list.append(nn.Parameter(torch.zeros(in_dim, self.expert_num), requires_grad=True))
                for i in range(self.expert_num):  #  Expert number
                    self.lora_A = nn.Linear(
                        self.in_features_,
                        self.r_,
                        bias=False,
                        dtype=self.dtype_,
                        device=self.device_,
                    )
                    self.lora_B = nn.Linear(
                        self.r_,
                        self.out_features_,
                        bias=False,
                        dtype=self.dtype_,
                        device=self.device_,
                    )
                    self.lora_A_list.append(self.lora_A)
                    self.lora_B_list.append(self.lora_B)

        self.use_dora_: bool = use_dora_
        self.magnitude_vector_: nn.Parameter = None

        self.reset_parameters(weight)

    def _get_weight_norm(self) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = dequantize_module_weight(self.base_layer_).to(self.dtype_)
        lora_weight = self.lora_B.weight @ self.lora_A.weight
        weight = weight + self.scaling_ * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def reset_parameters(
        self, weight: Tuple[torch.Tensor, torch.Tensor] = (None, None)
    ) -> None:
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(weight, Tuple)
        assert len(weight) == 2
        assert ((weight[0] is None) and (weight[1] is None)) or (
            isinstance(weight[0], torch.Tensor) and isinstance(weight[1], torch.Tensor)
        )

        if weight == (None, None):
            if self.initializer_ == "original":
                for i in range(len(self.lora_A_list)):
                    nn.init.kaiming_uniform_(self.lora_A_list[i].weight, a=math.sqrt(5))
                    # nn.init.kaiming_uniform_(self.lora_B_list[i].weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                for i in range(len(self.lora_A_list)):
                    nn.init.normal_(self.lora_A_list[i].weight, std=1 / self.r_)
                    # nn.init.normal_(self.lora_B_list[i].weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            for i in range(len(self.lora_B_list)):
                nn.init.zeros_(self.lora_B_list[i].weight)
        else:
            with torch.no_grad():
                for i in range(len(self.lora_A_list)):
                    self.lora_A_list[i].weight.copy_(weight[0])
                    self.lora_B_list[i].weight.copy_(weight[1])

        if self.use_dora_:
            self.magnitude_vector_ = nn.Parameter(
                self._get_weight_norm(), requires_grad=True
            )

    def apply_dora(
        self,
        residual: torch.Tensor,
        result_lora: torch.Tensor,
    ):
        weight_norm = self._get_weight_norm().detach()
        mag_norm_scale = (self.magnitude_vector_ / weight_norm).view(1, -1)
        return mag_norm_scale * residual + mag_norm_scale * result_lora
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.expert_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def lora_forward(
        self, residual: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        x = hidden_states.clone()

    
        x_re = x.mean(dim=1)
        
        
    
        gates, load = self.noisy_top_k_gating(x_re, self.training, self.router_list[self.task_id],
                                                self.w_noise_list[self.task_id])
        
    
        

        

        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= 1e-2 # # Todo

        ###
        nonzero_indices = torch.nonzero(gates)
        counter = Counter(nonzero_indices[:, 1].tolist())
        # for number, count in counter.items():
        #     if self.text_or_image == 'text':
        #         self.choose_map_text[ number] = self.choose_map_text[number] + count
        #     else:
        #         self.choose_map_image[number] = self.choose_map_image[number] + count


        dispatcher = SparseDispatcher(self.expert_num, gates)
        
        # print(self.lora_A_list.requires_grad_)
        # print(self.lora_A_list[0].weight.requires_grad)

        # exit()
        
        expert_inputs = dispatcher.dispatch(x.reshape(x.shape[0], -1))
        # for e in expert_inputs:
        #     print(e.shape)
        # exit()
        # print(expert_inputs[0].shape)
        # exit()
        expert_outputs = [self.lora_B_list[i](self.lora_A_list[i](self.dropout_(expert_inputs[i].view(expert_inputs[i].shape[0],
                            x.shape[1],x.shape[2]).to(x))))
            * self.scaling_ for i in range(self.expert_num)]
        

 
        i = 0
        while i < len(expert_outputs):
            if expert_outputs[i].shape[0] == 0 :
                expert_outputs.pop(i)
            else:
                expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0],-1)
                i += 1

        y = dispatcher.combine(expert_outputs)
        
    
        y = y.view(x.shape[0],x.shape[1],self.out_features_)

        # print()
        # result_lora = (
        #     self.lora_B(self.lora_A(self.dropout_(hidden_states.to(self.dtype_))))
        #     * self.scaling_
        # )
        result_lora = y

        
        # print(result_lora.shape)
        # print(residual.shape)
        # exit()
        if self.use_dora_:
            return self.apply_dora(residual, result_lora).to(hidden_states.dtype)
        else:
            return residual + result_lora.to(residual.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.base_layer_(hidden_states)
        return self.lora_forward(residual, hidden_states)
    

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # print(self._num_experts)
        # sort experts
        # print('gates', gates.shape) # 64, 22
        # [[0.0000, 0.0000, 0.5146, 0.4854, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.0000, 0.0000, 0.0000, 0.0000, 0.4666, 0.5334, 0.0000, 0.0000, 0.0000]]
        # print(torch.nonzero(gates).shape)  # torch.Size([128, 2])
        
        # print(gates)
        # print(torch.nonzero(gates))
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        


        # print(sorted_experts.shape, index_sorted_experts.shape) # torch.Size([128, 2]) torch.Size([128, 2])
        # [[0, 2],[0, 3],[1, 4],[1, 5]] sorted_experts 将feature和experts匹配上
        # [[1, 0],[0, 1],[2, 2],[3, 3]]

        # drop indices
        # print(sorted_experts)
        # print(index_sorted_experts)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # print(self._expert_index)

        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
   
        # print(self._batch_index)
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weight


        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
