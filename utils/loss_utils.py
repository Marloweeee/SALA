import torch
import numpy as np
import torch.nn.functional as F

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou

def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            # gi = coord_list[scale_ii][ii,0].long()
            # gj = coord_list[scale_ii][ii,1].long()
            # tx = coord_list[scale_ii][ii,0] - gi.float()
            # ty = coord_list[scale_ii][ii,1] - gj.float()
            gw = coord_list[scale_ii][ii,2]
            gh = coord_list[scale_ii][ii,3]

            anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            ## Get shape of gt box
            # gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # import pdb
            # pdb.set_trace()

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/best_grid), \
            x[1] / (args.anchor_imsize/best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii,0].long()
        gj = coord_list[best_scale][ii,1].long()
        tx = coord_list[best_scale][ii,0] - gi.float()
        ty = coord_list[best_scale][ii,1] - gj.float()
        gw = coord_list[best_scale][ii,2]
        gh = coord_list[best_scale][ii,3]
        tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3,0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii]//3][ii, best_n_list[ii]%3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
        gt_conf_list.append(target[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


def trans_vg_loss(batch_pred, batch_target):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses


def get_each_grid(feature_map_size:int,grid_size:int,normalized_coords):

    # 计算目标框在特征图中的位置
    target_boxes = normalized_coords*feature_map_size

    # 计算每个下采样后的网格与目标框的IoU
    iou_scores = []

    # 计算每个下采样后的网格与每个目标框的IoU
    for box in target_boxes:
        box = box.tolist()  # [x,y,w,h]
        iou_per_box = []
        for grid_x in range(0, feature_map_size, grid_size):
            for grid_y in range(0, feature_map_size, grid_size):
                grid_box = [grid_y, grid_x, grid_y + grid_size, grid_x + grid_size]

                # 计算交集的坐标
                intersection_x1 = max(box[0]-box[2]/2, grid_box[0])# x1=x-w/2
                intersection_y1 = max(box[1]-box[3]/2, grid_box[1])# y1=y-h/2
                intersection_x2 = min(box[0] + box[2]/2, grid_box[2])# x2=x+w/2
                intersection_y2 = min(box[1] + box[3]/2, grid_box[3])# y2=y+h/2

                # 计算交集面积和并集面积
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0,intersection_y2 - intersection_y1)
                iou_per_box.append(1 if intersection_area != 0 else -10000)

        # 将目标框与patch的IoU转化为tensor
        iou_tensor = torch.tensor(iou_per_box).float()
        softmax_scores = F.softmax(iou_tensor, dim=-1)
        softmax_scores /= torch.max(softmax_scores)

        # 将每个目标框的IoU得分存入列表
        iou_scores.append(softmax_scores)

    res = torch.stack(iou_scores)
    return res

def get_similarity(visu_src, txt_features, visu_token_mask, cross_modal=True):

        # First extract the positive image tokens mean value from visual features
        # N B C -> B C N
        visu_src=visu_src.permute(1,2,0)
        
        visu_feat=[]
        for b in range(visu_src.shape[0]):
            pos_visu_region=visu_src[b,:,visu_token_mask[b].bool()]# [B, C, N] -> [C,N_p]
            visu_feat.append(torch.mean(pos_visu_region,dim=-1))# [B,C]
            # visu_feat.append(torch.max(pos_visu_region, dim=-1)[0]+torch.mean(pos_visu_region,dim=-1))
        img_features=torch.stack(visu_feat)

        # normalize features
        img_features=img_features/img_features.norm(dim=1,keepdim=True)
        txt_features=txt_features/txt_features.norm(dim=1,keepdim=True)

        if cross_modal:
            "if cross_modal, calculate the similarity between image and text features"
            logits_per_img=img_features @ txt_features.t()
            logits_per_txt=logits_per_img.t()
            # return logits_per_img,logits_per_txt
        # else:
            "if union_modal, calculate the similarity between image and text features"
            logits_img_img=img_features @ img_features.t()
            logits_txt_txt=txt_features @ txt_features.t()
            return logits_per_img,logits_per_txt,logits_img_img,logits_txt_txt

def get_teacher_features_similarity(visual_teacher_features, text_teacher_features):

        # normalize features
        visu_teacher_features=visual_teacher_features/visual_teacher_features.norm(dim=1,keepdim=True)
        txt_teacher_features=text_teacher_features/text_teacher_features.norm(dim=1,keepdim=True)

        # calculate similarity under unionteacher features 
        softlabel_image_sim = F.cosine_similarity(visu_teacher_features.unsqueeze(1), visu_teacher_features.unsqueeze(0), dim=-1)
        softlabel_text_sim = F.cosine_similarity(txt_teacher_features.unsqueeze(1), txt_teacher_features.unsqueeze(0), dim=-1)

        return softlabel_image_sim, softlabel_text_sim


def PairedSampleAlignmentLoss(logits_per_img, logits_per_txt, tau, idx=None):

    # idx represents the positive sample index, if idx is None, it means all the samples are positive
    # sim_targets is a matrix with the same size as logits_per_img, with 1 on the diagonal and 0 elsewhere
    if idx is None:
        sim_targets=torch.eye(logits_per_img.shape[0]).to(logits_per_img.device)
    else:
        idx=idx.view(-1,1)
        pos_idx=torch.eq(idx,idx.t()).float().to(logits_per_img.device)
        sim_targets=pos_idx/torch.sum(pos_idx,dim=1,keepdim=True)

    # calculate contrastive loss
    loss_i2t= -torch.mean(F.log_softmax(logits_per_img/tau,dim=1)*sim_targets,dim=1).mean()
    loss_t2i= -torch.mean(F.log_softmax(logits_per_txt/tau,dim=1)*sim_targets,dim=1).mean()

    contrastive_loss=loss_i2t+loss_t2i
    return contrastive_loss

def KLAlignmentLoss(logits, soft_labels, tau, soft_labels_tau, use_loss='kl'):

    # softmax for both logits and soft_labels
    logit_inputs=F.log_softmax(logits/tau,dim=1)
    sim_targets=F.softmax(soft_labels/soft_labels_tau,dim=1)

    if use_loss == "kl":
        # KL divergence
        loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
    else:
        # Switch to the same loss as ContrastiveLoss, but sim_targets is soft
        loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()

    return loss

def LocalizationAlignmentLoss(visu_src, text_cls, visu_token_mask,tau):

    # normalize features
    visu_feat=F.normalize(visu_src,dim=-1)
    text_cls=F.normalize(text_cls,dim=-1)
    # transpose features
    visu_feat = visu_feat.permute(1, 2, 0)  # [B, C, N]
    text_cls = text_cls.unsqueeze(1)  # [B, 1, C]

    # calculate similiarity between text_cls and visual tokens 
    import math
    scale=math.sqrt(text_cls.shape[-1])
    logit=torch.matmul(text_cls,visu_feat)/(scale)# [B,1,N]
    logit=logit.squeeze()# [B, N]

    # loss iteration
    loss,B=[],visu_token_mask.shape[0]
    for b in range(B):

        pos_logit = logit[b][visu_token_mask[b].bool()]# [num_pos]
        neg_logit = logit[b][~(visu_token_mask[b].bool())]# [num_neg]
        neg_logit=torch.cat([neg_logit,torch.tensor([1e-30]).to(neg_logit.device)])

        pos_exp = torch.exp(pos_logit/tau)
        neg_exp = torch.exp(neg_logit/tau)

        single_img_loss = -torch.log(torch.div(pos_exp,pos_exp+torch.sum(neg_exp)))
        loss.append(torch.sum(single_img_loss)/torch.sum(visu_token_mask[b]))

    # calculate mean value of final loss
    avg_loss=torch.stack(loss).mean()

    return avg_loss

def SALA_Loss(text_cls, visu_src, patch_mask,text_teacher_features, visual_teacher_features, args):

    logits_per_img,logits_per_txt,logits_img_img,logits_txt_txt=get_similarity(visu_src, text_cls, patch_mask)
    softlabel_image_sim, softlabel_text_sim=get_teacher_features_similarity(text_teacher_features, visual_teacher_features)
    
    # CUSA loss
    psa_loss=PairedSampleAlignmentLoss(logits_per_img, logits_per_txt, args.psa_tau)
    csa_loss=KLAlignmentLoss(logits_per_img, softlabel_image_sim, args.csa_tau, args.ct_tau, use_loss='kl')
    csa_loss+=KLAlignmentLoss(logits_per_txt, softlabel_text_sim, args.csa_tau, args.ct_tau, use_loss='kl')
    usa_loss=KLAlignmentLoss(logits_img_img, softlabel_image_sim, args.usa_tau, args.ut_tau, use_loss='kl')
    usa_loss+=KLAlignmentLoss(logits_txt_txt, softlabel_text_sim, args.usa_tau, args.ut_tau, use_loss='kl')

    sa_loss=args.alpha*psa_loss+args.beta*csa_loss+args.gama*usa_loss

    # PTCL loss
    la_loss=LocalizationAlignmentLoss(visu_src, text_cls, patch_mask, args.la_tau)

    
    return sa_loss,la_loss

