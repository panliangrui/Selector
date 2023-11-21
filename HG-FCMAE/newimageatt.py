from train_our import get_patients_information, _neg_partial_log, get_val_ci
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from models.our import fusion_model_our
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default="luad", help="Cancer type")
    parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")
    parser.add_argument("--out_classes", type=int, default=512, help="Model out dimension")
    parser.add_argument("--drop_out_ratio", type=float, default=0.5, help="Drop_out_ratio")
    parser.add_argument("--train_use_type", type=list, default=['img', 'rna', 'cli'], help='train_use_type,Please keep the relative order of img, rna, cli')
    parser.add_argument("--train_use_type1", type=list, default=['img'],help='train_use_type,Please keep the relative order of img, rna, cli')
    parser.add_argument("--mix", action='store_true', default=False, help="mix mae")
    args, _ = parser.parse_known_args()
    return args



args = get_params()
#########datasets
if args.cancer_type == 'lihc':
    patients = joblib.load('../HGCN/LIHC/lihc_patients.pkl')
    sur_and_time = joblib.load('../HGCN/LIHC/lihc_sur_and_time.pkl')
    all_data = joblib.load('../HGCN/LIHC/lihc_data.pkl')
    seed_fit_split = joblib.load('../HGCN/LIHC/lihc_split.pkl')
elif args.cancer_type == 'lusc':
    patients = joblib.load('../HGCN/LUSC/lusc_patients.pkl')
    sur_and_time = joblib.load('../HGCN/LUSC/lusc_sur_and_time.pkl')
    all_data = joblib.load('../HGCN/LUSC/lusc_data.pkl')
    seed_fit_split = joblib.load('../HGCN/LUSC/lusc_split.pkl')
elif args.cancer_type == 'esca':
    patients = joblib.load('../HGCN/ESCA/esca_patients.pkl')
    sur_and_time = joblib.load('../HGCN/ESCA/esca_sur_and_time_154.pkl')
    all_data = joblib.load('../HGCN/ESCA/esca_data.pkl')
    seed_fit_split = joblib.load('../HGCN/ESCA/esca_split.pkl')
elif args.cancer_type == 'luad':
    patients = joblib.load('../HGCN/LUAD/luad_patients.pkl')
    sur_and_time = joblib.load('../HGCN/LUAD/luad_sur_and_time.pkl')
    all_data = joblib.load('../HGCN/LUAD/luad_data.pkl')
    seed_fit_split = joblib.load('../HGCN/LUAD/luad_split.pkl')
elif args.cancer_type == 'ucec':
    patients = joblib.load('../HGCN/UCEC/ucec_patients.pkl')
    sur_and_time = joblib.load('../HGCN/UCEC/ucec_sur_and_time.pkl')
    all_data = joblib.load('../HGCN/UCEC/ucec_data.pkl')
    seed_fit_split = joblib.load('../HGCN/UCEC/ucec_split.pkl')
elif args.cancer_type == 'kirc':
    patients = joblib.load('../HGCN/KIRC/kirc_patients.pkl')
    sur_and_time = joblib.load('../HGCN/KIRC/kirc_sur_and_time.pkl')
    all_data = joblib.load('../HGCN/KIRC/kirc_data.pkl')
    seed_fit_split = joblib.load('../HGCN/KIRC/kirc_split.pkl')

patient_sur_type, patient_and_time, kf_label = get_patients_information(patients, sur_and_time)


# kf = StratifiedKFold(n_splits=0, shuffle=True)
# for train_index in (len(kf_label)):
train_index = np.arange(len(patients))
train_data = np.array(patients)[train_index]

# fold_patients.append(train_data)
# fold_patients.append(val_data)
# fold_patients.append(test_data)
# seed_patients.append(fold_patients)



model2 = fusion_model_our(args, in_feats=1024,
                                          n_hidden=args.n_hidden,
                                          out_classes=args.out_classes,
                                          dropout=args.drop_out_ratio,
                                          train_type_num=len(args.train_use_type)).to(device)
model1 = torch.load('../results/LUAD/our/predict_model.pth')
model2.load_state_dict(model1,strict=False)

model2.eval()

lbl_pred_all = None
status_all = []
survtime_all = []
val_pre_time = {}
val_pre_time_img = {}
val_pre_time_rna = {}
val_pre_time_cli = {}
iter = 0

for i_batch, id in enumerate(train_data):

    graph = all_data[id].to(device)
    if args.train_use_type != None:
        use_type_eopch = args.train_use_type1
    else:
        use_type_eopch = graph.data_type
    out_pre, out_fea, out_att, _ = model2(graph, args.train_use_type1, use_type_eopch, mix=args.mix) #
    lbl_pred = out_pre[0]

    if id=='TCGA-49-4494':
        coord = out_att[0]
        scores_all = out_att[1]

        break


    survtime_all.append(patient_and_time[id])
    status_all.append(patient_sur_type[id])

    val_pre_time[id] = lbl_pred.cpu().detach().numpy()[0]

    if iter == 0 or lbl_pred_all == None:
        lbl_pred_all = lbl_pred
    else:
        lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])

    # iter += 1
    #
    # if 'img' in use_type_eopch:
    #     val_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
    # if 'rna' in use_type_eopch:
    #     val_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()
    # if 'cli' in use_type_eopch:
    #     val_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy()



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
# 定义 patch 的尺寸
patch_width = 256
patch_height = 256

# 定义图像尺寸
image_width = 2048
image_height = 2048

# 文件夹路径和每个 patch 的坐标和得分
folder_path = 'J:\\pytorch projects\\HGCN-main\\Features\\luad_features\\patch\\TCGA-49-4494-01Z-00-DX1.2bf10e85-465a-4b17-abd7-3565f0e538a5'
import json

# 读取WSI坐标文件
with open("../Features/luad_features/coord/TCGA-49-4494-01Z-00-DX1.2bf10e85-465a-4b17-abd7-3565f0e538a5.json", "r") as file:
    coordinates = json.load(file)

patch_coordinates = []

for patch in coordinates["coords"]:
    x1 = patch["col"]
    y1 = patch["row"]
    x2 = patch["x"]
    y2 = patch["y"]

    patch_coordinates.append((x2 , y2))

coords = np.array(patch_coordinates)

scores = np.array((scores_all[0]*1000).detach().cpu())

from heatmaps.heatmap_utils import drawHeatmap, compute_from_patches

import torchvision.models as models
import torch
import torch.nn as nn
from cut_and_pretrain.cut_and_pretrain import fully_connected
slide_path = 'J:\\LUAD_WSI\\luad_wsi\\TCGA-49-4494-01Z-00-DX1.2bf10e85-465a-4b17-abd7-3565f0e538a5.svs'


model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size= (1,1)))
num_ftrs = model.classifier.in_features
model_final = fully_connected(model.features, num_ftrs, 30)
model = model.to(device)
model_final.eval()
model_final = model_final.to(device)
model_final = nn.DataParallel(model_final)

model_final.load_state_dict(torch.load('../weights/KimiaNetPyTorchWeights.pth')) #KimiaNetPyTorchWeights.pth

# wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
# _, _, wsi_object = compute_from_patches(wsi_object=wsi_object,
# 											model=model,
# 											feature_extractor=feature_extractor,
# 											batch_size=exp_args.batch_size, **blocky_wsi_kwargs,
# 											attn_save_path=None, feat_save_path=h5_path,
# 											ref_scores=None)
#
# heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap='jet',
#                       alpha=0.4, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
#                       thresh=-1, patch_size=(512,512), convert_to_percentiles=True)
#
# heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))

