import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import matplotlib.pyplot as plt

from module.eeg_encoder.model import EEGProject
from module.projector import ProjectorLinear
from module.dataset import EEGPreImageDataset


def set_seed(seed: int):
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_and_model(args, device):
    """Build dataset/model, load checkpoint, forward a batch to get features."""
    dataset = EEGPreImageDataset(
        args.sub_id, args.eeg_data_dir, args.selected_channels,
        args.time_window, args.image_feature_dir, args.text_feature_dir,
        True,
        args.image_aug_feature_dir,
        True, False, None, False, True, False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Use latent_dim inferred from dataset image features
    latent_dim = dataset.image_features.shape[-1]
    model = EEGProject(
        feature_dim=latent_dim,
        eeg_sample_points=args.time_window[1] - args.time_window[0],
        channels_num=len(args.selected_channels)
    ).to(device)
    eeg_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)
    img_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)

    # Load weights
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    eeg_projector.load_state_dict(checkpoint['eeg_projector_state_dict'])
    img_projector.load_state_dict(checkpoint['img_projector_state_dict'])

    model.eval()
    eeg_projector.eval()
    img_projector.eval()

    # Note: keep last batch behavior to match original script
    with torch.no_grad():
        for batch in dataloader:
            eeg_batch = batch[0].to(device)
            image_feature_batch = batch[1].to(device)
            eeg_feature_batch = eeg_projector(model(eeg_batch))
            image_feature_batch = img_projector(image_feature_batch)

    eeg_features = eeg_feature_batch.cpu().numpy()
    image_features = image_feature_batch.cpu().numpy()

    if args.normalize:
        # Optional L2 normalization on features
        eeg_features = eeg_features / (np.linalg.norm(eeg_features, ord=2, axis=1, keepdims=True) + 1e-8)
        image_features = image_features / (np.linalg.norm(image_features, ord=2, axis=1, keepdims=True) + 1e-8)

    return eeg_features, image_features


def compute_similarity(eeg_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between EEG and image features."""
    return cosine_similarity(eeg_features, image_features)


def plot_similarity_matrix(similarity_matrix: np.ndarray, out_path: str, large: bool):
    """Plot and save the similarity heatmap."""
    if large:
        label_fontsize = 24
        ticks_fontsize = 20
    else:
        label_fontsize = 34
        ticks_fontsize = 28

    fig = plt.figure(figsize=(11, 10))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)

    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])

    # Draw heatmap
    img = ax.imshow(similarity_matrix, cmap='coolwarm', aspect='equal')

    # Colorbar
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=label_fontsize)

    # Labels
    ax.set_xlabel('Image Features', fontsize=label_fontsize)
    ax.set_ylabel('EEG Features', fontsize=label_fontsize)

    # Ticks (every 20)
    ax.set_xticks(np.arange(0, similarity_matrix.shape[1], 20))
    ax.set_xticklabels(np.arange(0, similarity_matrix.shape[1], 20), fontsize=ticks_fontsize)
    ax.set_yticks(np.arange(0, similarity_matrix.shape[0], 20))
    ax.set_yticklabels(np.arange(0, similarity_matrix.shape[0], 20), fontsize=ticks_fontsize)

    # Layout
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)

    # Save
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    # I/O and runtime
    parser.add_argument("--out", type=str, default="similarity_matrix.pdf")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--large", action="store_true", default=False)

    # Data/model config (aligned with draw_retrival_sample.py)
    parser.add_argument("--eeg_data_dir", type=str, default="data/things_eeg/preprocessed_eeg")
    parser.add_argument("--image_feature_dir", type=str, default="data/things_eeg/image_feature/RN50")
    parser.add_argument("--image_aug_feature_dir", nargs="+", default=["data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"])
    parser.add_argument("--text_feature_dir", type=str, default="")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--sub_id", nargs="+", type=int, default=[8])
    parser.add_argument("--selected_channels", nargs="+", default=['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'])
    parser.add_argument("--time_window", nargs=2, type=int, default=[0, 250])
    parser.add_argument("--checkpoint_path", type=str, default="intra-subjects_sub-08_checkpoint_last.pth")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading data and model...")
    eeg_features, image_features = load_data_and_model(args, device)
    
#   new_order = np.array([83, 1, 103, 138, 36, 139, 148, 140, 143, 104, 
#                      105, 2, 149, 84, 37, 106, 85, 38, 150, 107, 
#                      108, 39, 109, 31, 86, 151, 40, 152, 41, 153, 
#                      87, 42, 3, 4, 110, 154, 155, 43, 5, 111, 
#                      156, 112, 113, 114, 157, 32, 44, 45, 158, 46,
#                      47, 159, 33, 48, 49, 160, 50, 34, 88, 51, 
#                      52, 115, 6, 53, 7, 144, 161, 162, 8, 9, 
#                      54, 10, 55, 163, 89, 11, 116, 117, 164, 118, 
#                      56, 57, 119, 90, 91, 12, 13, 14, 15, 120, 
#                      58, 121, 122, 165, 166, 141, 16, 167, 168, 92, 
#                      59, 169, 170, 142, 123, 17, 171, 172, 60, 18, 
#                      19, 173, 61, 124, 93, 125, 174, 175, 126, 62, 
#                      176, 63, 64, 65, 66, 177, 20, 178, 21, 179, 
#                      67, 68, 22, 127, 69, 23, 24, 180, 128, 70, 
#                      71, 25, 72, 26, 129, 181, 73, 74, 145, 27, 
#                      130, 28, 182, 94, 183, 184, 75, 76, 77, 95, 
#                      29, 185, 186, 96, 97, 187, 146, 131, 188, 132,
#                      133, 98, 134, 78, 99, 189, 190, 191, 192, 193, 
#                      194, 195, 35, 79, 135, 136, 196, 100, 197, 30, 
#                      101, 137, 147, 198, 80, 81, 102, 199, 82, 200])

    print("Computing cosine similarity...")
    similarity_matrix = compute_similarity(eeg_features, image_features)
    print(similarity_matrix.shape)

    print("Plotting similarity matrix...")
    plot_similarity_matrix(similarity_matrix, args.out, args.large)

    print(f"Done! Saved to {args.out}")


if __name__ == "__main__":
    main()