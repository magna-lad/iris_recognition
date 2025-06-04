# implementation of DOI-> 10.1109/ACCESS.2020.2973433


import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import pickle
import time

# Enable GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_image_paths_and_labels(root_dir):
    image_paths = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    labels = []
    for cls_name in class_names:
        cls_path = os.path.join(root_dir, cls_name)
        for img_file in glob.glob(os.path.join(cls_path, '*.jpg')):
            image_paths.append(img_file)
            labels.append(class_to_idx[cls_name])
    return image_paths, labels, class_to_idx

def stratified_per_class_split(image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    class_to_imgs = defaultdict(list)
    for p, l in zip(image_paths, labels):
        class_to_imgs[l].append(p)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    rng = np.random.default_rng(seed)
    for l, imgs in class_to_imgs.items():
        imgs = np.array(imgs)
        idx = np.arange(len(imgs))
        rng.shuffle(idx)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]
        for i in train_idx:
            train_paths.append(imgs[i])
            train_labels.append(l)
        for i in val_idx:
            val_paths.append(imgs[i])
            val_labels.append(l)
        for i in test_idx:
            test_paths.append(imgs[i])
            test_labels.append(l)
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def remove_reflections(img):
    mask = cv2.inRange(img, 240, 255)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img

def segment_iris_hough(img):
    blurred = cv2.medianBlur(img, 7)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=img.shape[0]//8,
                               param1=50, param2=30, minRadius=20, maxRadius=img.shape[0]//2)
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        circles = sorted(circles, key=lambda x: x[2])
        if len(circles) >= 2:
            pupil = circles[0]
            iris = circles[-1]
            return pupil, iris
    h, w = img.shape
    return (w//2, h//2, h//8), (w//2, h//2, h//3)

def daugman_rubber_sheet(img, pupil, iris, rad_res=64, ang_res=256):
    h, w = img.shape
    x_p, y_p, r_p = pupil
    x_i, y_i, r_i = iris
    theta = np.linspace(0, 2*np.pi, ang_res)
    r = np.linspace(0, 1, rad_res)
    polar = np.zeros((rad_res, ang_res), dtype=np.uint8)
    for j in range(ang_res):
        for i in range(rad_res):
            r_frac = r[i]
            x = (1 - r_frac) * x_p + r_frac * x_i + ((1 - r_frac) * r_p + r_frac * r_i) * np.cos(theta[j])
            y = (1 - r_frac) * y_p + r_frac * y_i + ((1 - r_frac) * r_p + r_frac * r_i) * np.sin(theta[j])
            x, y = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
            polar[i, j] = img[y, x]
    return polar

def preprocess_single_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = remove_reflections(img)
        pupil, iris = segment_iris_hough(img)
        norm_img = daugman_rubber_sheet(img, pupil, iris, rad_res=64, ang_res=256)
        img = cv2.resize(norm_img, (128, 128))
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def parallel_preprocess(image_paths, num_processes=None):
    if num_processes is None:
        num_processes = min(cpu_count(), 8)
    
    print(f"Preprocessing {len(image_paths)} images using {num_processes} processes...")
    start_time = time.time()
    
    with Pool(num_processes) as pool:
        results = pool.map(preprocess_single_image, image_paths)
    
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
    
    valid_results = [(img, path) for img, path in zip(results, image_paths) if img is not None]
    return valid_results

def get_cached_dataset(image_paths, labels, cache_file='preprocessed_cache.pkl'):
    if os.path.exists(cache_file):
        print("Loading cached preprocessed data...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Loaded {len(cached_data['images'])} cached images")
        return cached_data['images'], cached_data['labels']
    
    print("Cache not found. Preprocessing images...")
    preprocessed_results = parallel_preprocess(image_paths)
    
    preprocessed_images = []
    valid_labels = []
    
    for i, (result, path) in enumerate(zip(*zip(*preprocessed_results))):
        if result is not None:
            preprocessed_images.append(result)
            original_idx = image_paths.index(path)
            valid_labels.append(labels[original_idx])
    
    data = {'images': preprocessed_images, 'labels': valid_labels}
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Cached {len(preprocessed_images)} processed images")
    return preprocessed_images, valid_labels

class OptimizedIrisDataset(Dataset):
    def __init__(self, preprocessed_images, labels):
        self.images = preprocessed_images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        label = self.labels[idx]
        return img, label

class TinyVGG(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(512*8*8, 16384)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16384, feature_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        features = F.relu(self.fc2(x))
        features = self.dropout2(features)
        logits = self.fc3(features)
        return features, logits

class TCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(TCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.register_buffer('centers', torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(self.centers, p=2, dim=1)
        batch_centers = centers_norm[labels]
        loss = ((features_norm - batch_centers) ** 2).sum(dim=1).mean() / 2
        
        with torch.no_grad():
            unique_labels = labels.unique()
            for lbl in unique_labels:
                mask = (labels == lbl)
                if mask.sum() > 0:
                    mean_feat = features_norm[mask].mean(dim=0)
                    self.centers[lbl] = (1 - self.alpha) * self.centers[lbl] + self.alpha * mean_feat
        return loss

def train_one_epoch_optimized(model, loader, optimizer, criterion_softmax, criterion_tcenter, lambda_tc, device, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            features, logits = model(imgs)
            loss_softmax = criterion_softmax(logits, labels)
            loss_tcenter = criterion_tcenter(features, labels)
            loss = loss_softmax + lambda_tc * loss_tcenter
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, loader, criterion_softmax, criterion_tcenter, lambda_tc, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                features, logits = model(imgs)
                loss_softmax = criterion_softmax(logits, labels)
                loss_tcenter = criterion_tcenter(features, labels)
                loss = loss_softmax + lambda_tc * loss_tcenter
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                feats, _ = model(imgs)
                feats = F.normalize(feats, p=2, dim=1)
            features.append(feats.cpu())
            labels.extend(lbls)
    features = torch.cat(features, dim=0).numpy()
    return features, np.array(labels)

def generate_pairs(features, labels, num_pairs=1000):
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_indices[lbl].append(idx)
    
    pos_scores, neg_scores = [], []
    rng = np.random.default_rng(42)
    
    for lbl, idxs in label_to_indices.items():
        if len(idxs) < 2:
            continue
        num_pairs_class = min(num_pairs, len(idxs) * (len(idxs) - 1) // 2)
        pairs = rng.choice(idxs, (num_pairs_class, 2), replace=False)
        for i, j in pairs:
            if i != j:
                sim = np.dot(features[i], features[j])
                pos_scores.append(sim)
    
    label_list = list(label_to_indices.keys())
    for _ in range(num_pairs):
        if len(label_list) >= 2:
            lbl1, lbl2 = rng.choice(label_list, 2, replace=False)
            i = rng.choice(label_to_indices[lbl1])
            j = rng.choice(label_to_indices[lbl2])
            sim = np.dot(features[i], features[j])
            neg_scores.append(sim)
    
    return np.array(pos_scores), np.array(neg_scores)

if __name__ == "__main__":
    root_dir = "/kaggle/input/iris-dataset/iris"
    
    print("Loading dataset...")
    image_paths, labels, class_to_idx = get_image_paths_and_labels(root_dir)
    
    min_images_per_class = 3
    class_counts = Counter(labels)
    filtered = [(p, l) for p, l in zip(image_paths, labels) if class_counts[l] >= min_images_per_class]
    if not filtered:
        raise ValueError("No classes with at least 3 images found. Check your dataset.")
    
    image_paths, labels = zip(*filtered)
    print(f"Dataset: {len(set(labels))} classes, {len(image_paths)} total images")
    
    cached_images, cached_labels = get_cached_dataset(image_paths, labels)
    
    (train_paths_idx, train_labels), (val_paths_idx, val_labels), (test_paths_idx, test_labels) = stratified_per_class_split(
        list(range(len(cached_images))), cached_labels)
    
    train_images = [cached_images[i] for i in train_paths_idx]
    val_images = [cached_images[i] for i in val_paths_idx]
    test_images = [cached_images[i] for i in test_paths_idx]
    
    num_classes = len(set(cached_labels))
    print(f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    train_dataset = OptimizedIrisDataset(train_images, train_labels)
    val_dataset = OptimizedIrisDataset(val_images, val_labels)
    test_dataset = OptimizedIrisDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = TinyVGG(num_classes=num_classes).to(device)
    criterion_softmax = nn.CrossEntropyLoss()
    criterion_tcenter = TCenterLoss(num_classes, 512, alpha=0.5).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion_tcenter.parameters()), 
                                lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    scaler = GradScaler()
    lambda_tc = 0.5
    
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch_optimized(model, train_loader, optimizer, 
                                              criterion_softmax, criterion_tcenter, lambda_tc, device, scaler)
        val_loss = validate_epoch(model, val_loader, criterion_softmax, criterion_tcenter, lambda_tc, device)
        
        scheduler.step()
        
        print(f'Epoch {epoch:3d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    print("Loading best model and evaluating...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    features, test_labels_array = extract_features(model, test_loader, device)
    pos_scores, neg_scores = generate_pairs(features, test_labels_array)
    
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_scores = np.concatenate([pos_scores, neg_scores])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('T-Center Iris Recognition ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f'Test ROC AUC: {roc_auc:.4f}')
        print(f'Positive pairs: {len(pos_scores)}, Negative pairs: {len(neg_scores)}')
    else:
        print("Not enough positive or negative pairs in the test set for ROC computation.")
