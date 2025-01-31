import torch


@torch.no_grad()
def greedy_rearrange(mask, grads):
    num_unpruned = int(mask.sum())
    num_pruned = mask.shape[0] - num_unpruned
    if num_unpruned == 0 or num_pruned == 0:
        return mask

    grads = grads.permute(1, 0).contiguous() # shape: [#heads/neurons, #mini_batches] #12,64
    grads_sq = grads.pow(2).sum(dim=1) # 12
    _, indicies = grads_sq.sort(descending=False) #fisher信息从小到大 升序
    indicies = indicies.tolist()

    # Greedy search
    masked_indicies = indicies[:num_pruned] 
    for index in indicies[num_pruned:]:
        masked_indicies.append(index)
        grad_vectors = grads[masked_indicies] 
        grad_sum = grad_vectors.sum(dim=0)#64 

        complement = grad_sum - grad_vectors 
        grad_sum_length = complement.pow(2).sum(dim=1) #组合最小
                                                               
        removed = grad_sum_length.argmin()#最小的，也就是移除它最小
        del masked_indicies[removed]

    new_mask = torch.ones_like(mask)
    new_mask[masked_indicies] = 0
    return new_mask


def rearrange_mask(mask, grads):
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    device = mask.device
    mask = mask.cpu()
    grads = grads.cpu()

    num_hidden_layers = mask.shape[0]
    for i in range(num_hidden_layers):
        mask[i] = greedy_rearrange(mask[i], grads[:, i, :]) # 64,12

    mask = mask.to(device)
    return mask
