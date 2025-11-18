import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import timm

# Set random seed untuk reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# Konfigurasi
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "results"

# Buat directory jika belum ada
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class IndonesianFoodDataset(Dataset):
    """
    Custom Dataset untuk Indonesian Food Classification.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path ke file CSV dengan annotations
            root_dir (str): Directory dengan semua images
            transform (callable, optional): Transform untuk di-apply ke sample
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load label mapping
        import json
        label_mapping_path = DATASET_DIR / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            self.label_to_idx = label_mapping['label_to_idx']
        else:
            # Create on-the-fly if not exists
            unique_labels = sorted(self.df['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.df.iloc[idx]["filename"]
        image = Image.open(img_name).convert("RGB")
        
        # Get label (handle both string and integer labels)
        label_value = self.df.iloc[idx]["label"]
        if isinstance(label_value, str):
            label = self.label_to_idx[label_value]
        else:
            label = int(label_value)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms():
    """
    Mendefinisikan transforms untuk training dan validation.

    Returns:
        dict: Dictionary berisi 'train' dan 'val' transforms
    """
    # ImageNet statistics untuk normalisasi
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # Random erasing for regularization
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    return data_transforms


def create_dataloaders(batch_size=32):
    """
    Membuat DataLoader untuk training dan validation.

    Args:
        batch_size (int): Ukuran batch

    Returns:
        dict: Dictionary berisi 'train' dan 'val' DataLoader
    """
    data_transforms = get_data_transforms()

    # Buat datasets
    train_dataset = IndonesianFoodDataset(
        csv_file=DATASET_DIR / "train_split.csv",
        root_dir=DATASET_DIR / "train",
        transform=data_transforms["train"],
    )

    val_dataset = IndonesianFoodDataset(
        csv_file=DATASET_DIR / "val_split.csv",
        root_dir=DATASET_DIR / "train",
        transform=data_transforms["val"],
    )

    # Buat dataloaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
    }

    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    return dataloaders, dataset_sizes


def create_model(num_classes=5, pretrained=True):
    """
    Membuat DeiT model dengan pre-trained weights.

    Args:
        num_classes (int): Jumlah kelas output
        pretrained (bool): Apakah menggunakan pre-trained weights

    Returns:
        nn.Module: DeiT model
    """
    print("\n" + "=" * 60)
    print("CREATING DEIT MODEL")
    print("=" * 60)

    # Load pre-trained DeiT Tiny model dari timm
    model = timm.create_model(
        "deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes
    )

    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Model created successfully!")
    print(f"  Model: DeiT Tiny Patch16 224")
    print(f"  Pre-trained: {pretrained}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")

    return model


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=20,
    device="cuda",
    patience: int = 3,
):
    """
    Training loop untuk model.

    Args:
        model: PyTorch model
        dataloaders: Dictionary berisi train dan val DataLoader
        dataset_sizes: Dictionary berisi ukuran train dan val dataset
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Jumlah epoch
        device: Device untuk training (cuda/cpu)

    Returns:
        tuple: (model, history)
    """
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    model = model.to(device)

    # History untuk tracking metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_acc = 0.0
    best_model_wts = None
    early_stop_counter = 0
    stop_training = False

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Setiap epoch memiliki training dan validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Progress bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}")

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass hanya di training
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistik
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})

            # Learning rate scheduler step (setelah epoch)
            if phase == "train":
                scheduler.step()

            # Hitung epoch loss dan accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Simpan metrics ke history
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
                history["learning_rates"].append(optimizer.param_groups[0]["lr"])
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

            # Simpan best model dan cek early stopping
            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    early_stop_counter = 0
                    print(f"  → New best model! Validation Acc: {best_acc:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"  (No improvement) Early-stop counter: {early_stop_counter}/{patience}")
                    if early_stop_counter >= patience:
                        print(f"  → Early stopping triggered (patience={patience}).")
                        stop_training = True
                        break

        if stop_training:
            break

        time_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Acc: {best_acc:.4f}")
    print("=" * 60)

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def save_model_and_history(model, history, model_name="deit"):
    """
    Menyimpan model weights dan training history.

    Args:
        model: Trained model
        history: Training history
        model_name: Nama untuk file output
    """
    # Simpan model weights
    model_path = MODEL_DIR / f"{model_name}_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Simpan history
    history_path = OUTPUT_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"✓ Training history saved to: {history_path}")


def main():
    """
    Fungsi utama untuk menjalankan training DeiT.
    """
    print("\n" + "=" * 60)
    print("DEIT TRAINING")
    print("Indonesian Food Classification")
    print("=" * 60)

    # Hyperparameters
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 10 
    LEARNING_RATE = 5e-6 
    WEIGHT_DECAY = 0.1 

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # Buat dataloaders
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    dataloaders, dataset_sizes = create_dataloaders(batch_size=BATCH_SIZE)
    print(f"\n✓ Data loaded successfully!")
    print(f"  Training samples: {dataset_sizes['train']}")
    print(f"  Validation samples: {dataset_sizes['val']}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Buat model
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)

    # Loss function, optimizer, dan scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Loss function: CrossEntropyLoss")

    # Training (with early stopping)
    model, history = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        patience=3,
    )

    # Simpan model dan history
    save_model_and_history(model, history, model_name="deit")

    print("\n" + "=" * 60)
    print("DEIT TRAINING COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Evaluate both models: python src/evaluate.py")
    print("  2. Visualize results: python src/visualize.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
