import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ISBIVesselDataset3D(Dataset):
    def __init__(self, case_dirs, depth=16, img_size=(256, 256), augment=False,
                 rotations=(0, 90, 180, 270)):
        self.case_dirs = [Path(d) for d in case_dirs]
        self.depth = depth
        self.img_size = img_size
        self.augment = augment
        self.rotations = rotations if augment else (0,)

        # Construit la liste d'index (case_dir, angle)
        self.index = [
            (case_dir, angle)
            for case_dir in self.case_dirs
            for angle in self.rotations
        ]

    def __len__(self):
        return len(self.index)

    def _find_video(self, case_dir):
        avis = list(case_dir.glob("*.avi"))
        if not avis:
            raise FileNotFoundError(f"Aucun .avi dans {case_dir}")
        return avis[0]

    def _load_video_as_volume(self, avi_path):
        cap = cv2.VideoCapture(str(avi_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.img_size[1], self.img_size[0]))
            frames.append(gray)
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"Impossible de lire la vidéo {avi_path}")
        frames = self._sample_frames(frames, self.depth)
        volume = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return volume  # (D, H, W)

    def _sample_frames(self, frames, target_depth):
        n = len(frames)
        if n <= target_depth:
            return frames
        indices = np.linspace(0, n - 1, target_depth, dtype=int)
        return [frames[i] for i in indices]

    def _load_mask(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return mask  # (H, W)

    def _rotate_volume(self, volume, angle):
        if angle == 0:
            return volume
        k = angle // 90
        return np.rot90(volume, k=k, axes=(1, 2)).copy()  # (D, H, W)

    def _rotate_mask(self, mask, angle):
        if angle == 0:
            return mask
        k = angle // 90
        return np.rot90(mask, k=k, axes=(0, 1)).copy()  # (H, W)

    def __getitem__(self, idx):
        case_dir, angle = self.index[idx]
        avi_path = self._find_video(case_dir)
        mask_path = case_dir / "manual" / "forceMaskVessel.png"

        volume = self._load_video_as_volume(avi_path)   # (D, H, W)
        mask2d = self._load_mask(mask_path)              # (H, W)

        volume = self._rotate_volume(volume, angle)
        mask2d = self._rotate_mask(mask2d, angle)

        mask3d = np.stack([mask2d] * self.depth, axis=0)  # (D, H, W)

        volume = torch.from_numpy(volume).unsqueeze(0)   # (1, D, H, W)
        mask3d = torch.from_numpy(mask3d).unsqueeze(0)    # (1, D, H, W)

        return {"pixel_values": volume, "labels": mask3d}


def make_isbi_dataloaders(dataset_root, depth=16, img_size=(256, 256),
                          train_ratio=0.8, batch_size=2, num_workers=0, seed=42,
                          augment_train=True, rotations=(0, 90, 180, 270)):
    dataset_root = Path(dataset_root)
    case_dirs = sorted([d for d in dataset_root.iterdir()
                        if d.is_dir() and list(d.glob("*.avi"))])
    if not case_dirs:
        raise FileNotFoundError(f"Aucun cas trouvé dans {dataset_root}")

    print(f"{len(case_dirs)} cas trouvés")

    n_train = max(1, int(len(case_dirs) * train_ratio))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(len(case_dirs)))
    train_dirs = [case_dirs[i] for i in shuffled[:n_train]]
    val_dirs   = [case_dirs[i] for i in shuffled[n_train:]]

    train_ds = ISBIVesselDataset3D(
        train_dirs, depth=depth, img_size=img_size,
        augment=augment_train, rotations=rotations
    )
    val_ds = ISBIVesselDataset3D(
        val_dirs, depth=depth, img_size=img_size,
        augment=False  # pas d'augmentation sur la validation
    )

    print(f"Train: {len(train_dirs)} cas -> {len(train_ds)} échantillons (x{len(rotations) if augment_train else 1})")
    print(f"Val:   {len(val_dirs)} cas -> {len(val_ds)} échantillons")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl