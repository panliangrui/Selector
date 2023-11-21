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



model = fusion_model_our(args, in_feats=1024,
                                          n_hidden=args.n_hidden,
                                          out_classes=args.out_classes,
                                          dropout=args.drop_out_ratio,
                                          train_type_num=len(args.train_use_type)).to(device)
model1 = torch.load('../results/LUAD/our/predict_model.pth')
model.load_state_dict(model1,strict=False)

model.eval()

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
    out_pre, out_fea, out_att, _ = model(graph, args.train_use_type1, use_type_eopch, mix=args.mix) #
    lbl_pred = out_pre[0]

    if id=='TCGA-49-4494':
        coord = out_att[0]
        coord1 = out_att[1]

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
    data = json.load(file)

# 解析坐标数据
patch_coordinates = []

for patch in data["coords"]:
    x1 = patch["col"]
    y1 = patch["row"]
    x2 = patch["x"]
    y2 = patch["y"]

    patch_coordinates.append((x1, y1, x2, y2))

# 输出坐标信息
for coordinates in patch_coordinates:
    print(coordinates)

# coordinates = [(0, 0), (256, 0), (512, 0), ...]  # 每个 patch 的左上角坐标
scores = coord1[0]*1000  # 对应每个 patch 的得分

# 创建一个空白图像作为热力图的底图
heatmap = np.zeros((image_height, image_width))

# 遍历每个 patch，并将得分映射到热力图中
for coord, score in zip(patch_coordinates, scores):
    col, row, x, y = coord

    # 计算当前 patch 在热力图中的范围
    patch_start_x = x
    patch_end_x = x + patch_width
    patch_start_y = y
    patch_end_y = y + patch_height

    # 将 score 移动到 CPU 上，并使用 detach() 方法分离计算图
    score = score.detach().cpu()

    # 根据得分将对应区域设置为对应强度
    heatmap[patch_start_y:patch_end_y, patch_start_x:patch_end_x] = score.numpy()

# 将热力图进行可视化
# 定义自定义颜色映射
num_colors = 100  # 颜色数量
colors = [(0, 0, i/num_colors) for i in range(num_colors)]  # 蓝色到红色的渐变
cmap = ListedColormap(colors)


plt.imshow(heatmap, cmap=cmap)
plt.colorbar()
plt.show()



# survtime_all = np.asarray(survtime_all)
# status_all = np.asarray(status_all)
# #     print(lbl_pred_all,survtime_all,status_all)
# loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
# loss = loss_surv
#
# val_ci_ = get_val_ci(val_pre_time, patient_and_time, patient_sur_type)
# val_ci_img_ = 0
# val_ci_rna_ = 0
# val_ci_cli_ = 0
#
# if 'img' in args.train_use_type:
#     val_ci_img_ = get_val_ci(val_pre_time_img, patient_and_time, patient_sur_type)
# if 'rna' in args.train_use_type:
#     val_ci_rna_ = get_val_ci(val_pre_time_rna, patient_and_time, patient_sur_type)
# if 'cli' in args.train_use_type:
#     val_ci_cli_ = get_val_ci(val_pre_time_cli, patient_and_time, patient_sur_type)
