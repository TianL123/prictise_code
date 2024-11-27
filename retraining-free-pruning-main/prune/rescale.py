from tqdm import tqdm
import torch

from utils.linalg import lsmr_cupy_solver
from utils.arch import (
    get_layers,
    get_mha_proj,
    get_ffn2,
    hijack_input,
    MaskNeurons,
    remove_padding,
    collect_layer_inputs,
)


@torch.no_grad()
def get_mha_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
):
    num_attention_heads = config.num_attention_heads #12
    hidden_size = config.hidden_size #768
    attention_head_size = int(hidden_size / num_attention_heads) #64

    nonzero_heads = student_head_mask[layer_idx].nonzero().flatten()
    num_nonzero_heads = nonzero_heads.shape[0] #12

    layer = get_layers(model)[layer_idx]
    mha_proj = get_mha_proj(model, layer_idx)#attention.output
    weights_per_head = mha_proj.dense.weight.t().view(num_attention_heads, -1, hidden_size)#12,64,768
    weights_per_head = weights_per_head.index_select(dim=0, index=nonzero_heads)#12,64,768

    inputs = []
    handle = hijack_input(mha_proj, inputs)

    ATA = torch.zeros(num_nonzero_heads + 1, num_nonzero_heads + 1).cuda()#13,13
    ATB = torch.zeros(num_nonzero_heads + 1).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        attention_mask = (teacher_batch[1] == 0)
        student_batch[2] = student_head_mask[layer_idx].view(1, -1, 1, 1)#1,12,1,1

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            layer(*teacher_batch)
        hidden_states, input_tensor = inputs.pop(0)# hidden_states input_tensor :32,72,768 
        teacher_output = mha_proj.dense(hidden_states) + input_tensor #32,72,768
        teacher_output = remove_padding(teacher_output, attention_mask) # 1680,768

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            layer(*student_batch)
        hidden_states, input_tensor = inputs.pop(0)
        hidden_states = remove_padding(hidden_states, attention_mask)#1680,768
        input_tensor = remove_padding(input_tensor, attention_mask) #1680,768

        hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)#1680,12,64
        hidden_states = hidden_states.permute(1, 0, 2)#12,1680,64
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_heads)#根据head_mask选择 12,1680,64

        outputs_per_head = hidden_states @ weights_per_head #12,1680,64   12,64,768  ->12,1680,768
        outputs_per_head = outputs_per_head.view(num_nonzero_heads, -1)#12,1680*768

        A = outputs_per_head.t()#1290240,12
        A = torch.cat([A, torch.ones(A.shape[0], 1).cuda()], dim=1)#
        B = teacher_output - mha_proj.dense.bias - input_tensor #1680,768
        B = B.flatten()

        ATA += A.t() @ A # 13,13
        ATB += A.t() @ B

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def get_ffn_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
    cls_only=False,
):
    layer = get_layers(model)[layer_idx] #根据idx得到layer
    ffn2 = get_ffn2(model, layer_idx) #ffn.output
    weights_per_neuron = ffn2.dense.weight.t()#[3072,768] 输出层的权重

    nonzero_neurons = student_neuron_mask[layer_idx].nonzero().flatten() #3072  把非零元素取出来 1956
    num_neurons = nonzero_neurons.shape[0] #1956
    weights_per_neuron = weights_per_neuron.index_select(dim=0, index=nonzero_neurons) #在dim=0维度上选非零的掩码权重 1956 768 #1956,1,768
    W = weights_per_neuron @ weights_per_neuron.t() #1956 1956

    inputs = []
    handle = hijack_input(ffn2, inputs) #这里把列表传进去

    ATA = torch.zeros(num_neurons, num_neurons).cuda() #初始化ATA  ATB矩阵
    ATB = torch.zeros(num_neurons).cuda()

    model.eval() #评估模式
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs): #batch[2]:1,12,1,1这个输入时embediing之后的输出也是第一个layer的输入
        attention_mask = (teacher_batch[1] == 0)#[32, 1, 1, 72]
        student_batch[2] = student_head_mask[layer_idx].view(1, -1, 1, 1) #1,12,1,1

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            layer(*teacher_batch)
        hidden_states, input_tensor = inputs.pop(0)#32,72,3072  32,72,768 这里没有掩码输出
        teacher_output = ffn2.dense(hidden_states) + input_tensor #残差输出   ffn2.dense 32*72*3072 * 3072*768 
        if cls_only:
            teacher_output = teacher_output[:, 0, :]
        else:
            teacher_output = remove_padding(teacher_output, attention_mask)#1670,768如果不移除padding# 32,72,768 ->  1670,768

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            layer(*student_batch)
        hidden_states, input_tensor = inputs.pop(0) #32,83,3072  32,83,768
        if cls_only:
            hidden_states = hidden_states[:, 0, :]
            input_tensor = input_tensor[:, 0, :]
        else:
            hidden_states = remove_padding(hidden_states, attention_mask) #32*72,3072    1680,3072    
            input_tensor = remove_padding(input_tensor, attention_mask) #1680，768            

        hidden_states = hidden_states.t()#3072,1680  中间层的输出  残差输入         
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_neurons)#1956,1680 T*N   根据学生ffn掩码 得到  1952,1680,1
 
        ATA += W * (hidden_states @ hidden_states.t() ) #1956,1956                    

        B = teacher_output - ffn2.dense.bias - input_tensor #原始残差输出-输入  1744,768

        ATB += (hidden_states.unsqueeze(1) @ (weights_per_neuron @ B.t()).unsqueeze(2)).squeeze() #1956,1,1744
        #1956,768    1956,1744  1956,1744,1 #hidden_states.shape 1956,1744 非零通道N，TD   #weights_per_neuron:1956,768
    handle.remove()
    return ATA, ATB


@torch.no_grad()
def rescale_mask(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    dataloader,
    classification_task=False,
):
    num_hidden_layers = config.num_hidden_layers #12
    rescaled_head_mask = student_head_mask.clone() #修改副本不会影响原来
    rescaled_neuron_mask = student_neuron_mask.clone()

    for layer_idx in tqdm(range(num_hidden_layers)):
        teacher_inputs = collect_layer_inputs(
            model,
            teacher_head_mask,
            teacher_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else teacher_inputs,
        )
        student_inputs = collect_layer_inputs(
            model,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else student_inputs,
        )

        if torch.count_nonzero(student_head_mask[layer_idx]) != 0 and layer_idx != 0:
            ATA, ATB = get_mha_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            scale_factor = scale_factor[:-1]
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_heads = rescaled_head_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_heads, scale_factor):
                rescaled_head_mask[layer_idx][index] *= scale

        if torch.count_nonzero(student_neuron_mask[layer_idx]) != 0:
            cls_only = classification_task and (layer_idx == num_hidden_layers - 1)
            ATA, ATB = get_ffn_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
                cls_only=cls_only,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_neurons, scale_factor):
                rescaled_neuron_mask[layer_idx][index] *= scale

    return rescaled_head_mask, rescaled_neuron_mask
