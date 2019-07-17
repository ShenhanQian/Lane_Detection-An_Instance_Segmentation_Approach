import torch
import torch.nn.functional as F


def discriminative_loss_single(embedding, inst_label, delta_v, delta_d, param_var, param_dist, param_reg):
    c, h, w = embedding.size()
    if inst_label.is_cuda == True:
        device = 'cuda'
    else:
        device = 'cpu'

    # extract binary mask from instance label map for each class
    num_classes = len(torch.unique(inst_label)) - 1
    h, w = inst_label.shape
    inst_masks = torch.zeros(num_classes, h, w).byte().to(device)

    for idx, label in enumerate(torch.unique(inst_label)):
        if label == 0:  # deal with backgroud
            continue
        else:
            inst_masks[idx-1] = (inst_label == label)

    embeddings = []
    # select embedding pixel with inst_label
    for i in range(num_classes):
        feature = torch.transpose(torch.masked_select(embedding, inst_masks[i, :, :]).view(c, -1), 0, 1)
        embeddings.append(feature)

    centers = []
    for feature in embeddings:
        center = torch.mean(feature, dim=0, keepdim=True)
        centers.append(center)

    # intra-embedding loss within a class
    loss_var = torch.Tensor([0.0]).to(device)
    for feature, center in zip(embeddings, centers):
        dis = torch.norm(feature - center, 2, dim=1) - delta_v
        dis = F.relu(dis)
        loss_var += torch.mean(dis)
    loss_var /= num_classes

    if num_classes == 1:
        return loss_var, loss_var, torch.zeros(1)  # .to(device)

    # inter-class loss
    centers = torch.cat(centers, dim=0)
    A = centers.repeat(1, num_classes).view(-1, c)
    B = centers.repeat(num_classes, 1)
    distance = torch.norm(A - B, 2, dim=1).view(num_classes, num_classes)

    # select pair wise distance from distance matrix
    eye = torch.eye(num_classes).to(device)
    pair_distance = torch.masked_select(distance, eye == 0)

    pair_distance = delta_d - pair_distance
    pair_distance = F.relu(pair_distance)
    loss_dist = torch.mean(pair_distance).view(-1)

    # regularization loss
    loss_reg = torch.mean(torch.norm(centers, 2, dim=1)).view(-1)

    loss = param_var * loss_var + param_dist * loss_dist + param_reg * loss_reg
    return loss, loss_var, loss_dist, loss_reg


def discriminative_loss(embedding_batch, label_batch,
                        delta_v=0.5,  # 0.5
                        delta_d=1.5,
                        param_var=1.0,
                        param_dist=1.0,
                        param_reg=0.001):

    loss, loss_v, loss_d, loss_r = 0, 0, 0, 0
    for embedding, inst_lbl in zip(embedding_batch, label_batch):

        _loss, _loss_v, _loss_d, _loss_r = \
            discriminative_loss_single(embedding, inst_lbl, delta_v, delta_d, param_var, param_dist, param_reg)

        loss += _loss
        loss_v += _loss_v
        loss_d += _loss_d
        loss_r += _loss_r

    return loss, loss_v, loss_d, loss_r


if __name__ == '__main__':

    embedding_batch = torch.rand(2, 4, 256, 512)
    inst_lbl_batch = ((torch.rand(2, 256, 512)) * 9 + 1).byte()

    loss, loss_v, loss_d, loss_r = discriminative_loss(embedding_batch, inst_lbl_batch)
