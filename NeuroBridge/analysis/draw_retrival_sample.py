import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image

from module.eeg_encoder.model import EEGProject
from module.projector import ProjectorLinear
from module.dataset import EEGPreImageDataset


def load_data_and_model(args, device):
    # Configuration
    eeg_data_dir = args.eeg_data_dir
    image_feature_dir = args.image_feature_dir
    image_aug_feature_dir = args.image_aug_feature_dir
    text_feature_dir = args.text_feature_dir
    feature_dim = args.feature_dim
    sub_id = args.sub_id
    selected_channels = args.selected_channels
    time_window = args.time_window
    checkpoint_path = args.checkpoint_path

    dataset = EEGPreImageDataset(
        sub_id, eeg_data_dir, selected_channels,
        time_window, image_feature_dir, text_feature_dir,
        True,
        image_aug_feature_dir,
        True, False, None, False, True, False
    )
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)

    latent_dim = dataset.image_features.shape[-1] # default: 1024
    model = EEGProject(feature_dim=latent_dim, eeg_sample_points=time_window[1] - time_window[0], channels_num=len(selected_channels)).to(device)
    eeg_projector = ProjectorLinear(latent_dim, feature_dim).to(device)
    img_projector = ProjectorLinear(latent_dim, feature_dim).to(device)

    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    eeg_projector.load_state_dict(checkpoint['eeg_projector_state_dict'])
    img_projector.load_state_dict(checkpoint['img_projector_state_dict'])

    model.eval()
    eeg_projector.eval()
    img_projector.eval()

    with torch.no_grad():
        for batch in dataloader:
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            eeg_feature_batch = model(eeg_batch)
            eeg_feature_batch = eeg_projector(eeg_feature_batch)
            image_feature_batch = img_projector(image_feature_batch)

    eeg_features = eeg_feature_batch.cpu().numpy()
    image_features = image_feature_batch.cpu().numpy()

    return eeg_features, image_features


def build_image_list(args):
    # Configuration
    image_dir = args.image_dir
    
    image_classes = sorted(os.listdir(image_dir))
    image_list = []
    for cls in image_classes:
        cls_path = os.path.join(image_dir, cls)
        for f in sorted(os.listdir(cls_path)):
            image_list.append(os.path.join(cls_path, f))
    return image_list


def draw_grid(image_list, random_image_indices, random_topk_indices, out_path, k):
    rows = len(random_image_indices)
    cols = k + 1   # 1 (Seen) + k (Top-k)

    fontsize = 28
    col_titles = ["Seen"] + [f"Top {i}" for i in range(1, k + 1)]

    gs = GridSpec(rows, cols + 1, width_ratios=[1, 0.1] + [1] * (cols - 1), wspace=0, hspace=0)
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(rows):
        query_idx = random_image_indices[i]
        query_img = image_list[query_idx]

        for j in range(cols):
            img_path = query_img if j == 0 else image_list[random_topk_indices[i, j - 1]]
            grid_j = j if j == 0 else j + 1
            ax = fig.add_subplot(gs[i, grid_j])

            img = Image.open(img_path).resize((224, 224))
            ax.imshow(img, interpolation='none')

            # 如果检索命中查询图，加框
            if j != 0 and img_path == query_img:
                rect = Rectangle((0, 0), 224, 224, linewidth=8, edgecolor='#B40046', facecolor='none')
                ax.add_patch(rect)

            ax.axis("off")

    # 设置标题
    for j in range(cols):
        grid_j = j if j == 0 else j + 1
        ax = fig.add_subplot(gs[0, grid_j])
        ax.set_title(col_titles[j], fontsize=fontsize, weight='bold' if j == 0 else 'normal')
        ax.axis("off")

    plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out", type=str, default="retrieval_grid.pdf")
    parser.add_argument("--eeg_data_dir", type=str, default="data/things_eeg/preprocessed_eeg")
    parser.add_argument("--image_feature_dir", type=str, default="data/things_eeg/image_feature/RN50")
    parser.add_argument("--image_aug_feature_dir", nargs="+", default=["data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"])
    parser.add_argument("--text_feature_dir", type=str, default="")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--sub_id", nargs="+", type=int, default=[8])
    parser.add_argument("--selected_channels", nargs="+", default=['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'])
    parser.add_argument("--time_window", nargs=2, type=int, default=[0, 250])
    parser.add_argument("--checkpoint_path", type=str, default="intra-subjects_sub-08_checkpoint_last.pth")
    parser.add_argument("--image_dir", type=str, default="data/things_eeg/image_set/test_images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed all RNGs for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("Loading data and model...")
    eeg_features, image_features = load_data_and_model(args, device)

    print("Computing cosine similarity...")
    sim = cosine_similarity(eeg_features, image_features)

    # 获取 top-k
    k = args.topk
    topk_indices = np.argsort(sim, axis=1)[:, -k:][:, ::-1]

    # 随机 sample
    num_samples = args.num_samples
    random_image_indices = random.sample(range(image_features.shape[0]), num_samples)
    random_topk_indices = topk_indices[random_image_indices]

    print("Building image list...")
    image_list = build_image_list(args)

    print("Drawing grid...")
    draw_grid(image_list, random_image_indices, random_topk_indices,
              args.out, k)

    print(f"Done! Saved to {args.out}")


if __name__ == "__main__":
    main()
