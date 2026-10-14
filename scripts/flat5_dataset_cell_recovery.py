# Flat 5-class Dataset/DataLoader cell recovery.
# Run after the notebook import/configuration cell with %run -i.

class Flat5ClassDataset(torch.utils.data.Dataset):
    """Map seven source labels to 5 flat classes, retaining every bounding box."""

    class_names = CLASS_NAMES
    class_colors = CLASS_COLORS

    def __init__(self, base_dataset, require_labels=False, max_crop_attempts=32):
        self.base_dataset = base_dataset
        self.require_labels = bool(require_labels)
        self.max_crop_attempts = int(max_crop_attempts)
        if self.max_crop_attempts < 1:
            raise ValueError('max_crop_attempts must be positive')
        self.record_indices = list(range(len(base_dataset)))
        if self.require_labels:
            if base_dataset.four_crop or not base_dataset.random_crop:
                raise ValueError('require_labels is intended for random-crop training only')
            # 원본에 유효한 라벨이 전혀 없으면 재추출해도 소용없으므로 제외합니다.
            self.record_indices = [
                i for i in self.record_indices
                if len(base_dataset._get_resized_pair(i)[1]) > 0
            ]
            if not self.record_indices:
                raise ValueError('유효한 라벨이 있는 학습 원본이 없습니다.')
        self.excluded_empty_records = len(base_dataset) - len(self.record_indices)

    def __len__(self):
        return len(self.record_indices)

    def __getitem__(self, index):
        record_index = self.record_indices[index]
        attempts = self.max_crop_attempts if self.require_labels else 1
        for _ in range(attempts):
            image, classes, boxes, indices = self.base_dataset[record_index]
            if not self.require_labels or classes.numel() > 0:
                break
        else:
            # 무한 재시도를 피하고 라벨/crop 문제를 명시적으로 알립니다.
            raise RuntimeError(
                f'원본 index={record_index}: {attempts}회 crop 후에도 라벨이 없습니다. '
                '원본 라벨 좌표와 crop 설정을 확인하세요.'
            )
        mapping = classes.new_tensor(SOURCE_TO_FLAT)
        flat_classes = mapping[classes.long()]
        return image, flat_classes, boxes, indices

    @property
    def cache_memory_bytes(self):
        return self.base_dataset.cache_memory_bytes


def collate_fn(batch):
    return IHCHierarchicalDataset.collate_fn(batch)


input_size = 512
resize_size = 1024
data_root = '../../data/mediautoAI_PDL1_Stomach/'
VALIDATION_SLIDE_COUNT = 3

records = discover_records(data_root)
train_records, val_records = split_records_by_slide_balanced(
    records, val_slide_count=VALIDATION_SLIDE_COUNT
)
train_slides = {slide_id_from_record(record) for record in train_records}
val_slides = {slide_id_from_record(record) for record in val_records}
assert train_slides.isdisjoint(val_slides)

# 80%는 이미지에 존재하는 원본 클래스 중 하나를 동일 가중치로 선택합니다.
# 나머지는 일반 random crop입니다. 현재 flat 4-class 설정과 동일합니다.
CLASS_AWARE_CROP_PROBABILITY = 0.8
CROP_CLASS_WEIGHTS = [1, 1, 1, 1, 1, 1, 1]

train_base = IHCHierarchicalDataset(
    train_records, input_size=input_size, resize_size=resize_size,
    augment=True, random_crop=True, cache_images=True,
    class_aware_crop_probability=CLASS_AWARE_CROP_PROBABILITY,
    class_sampling_weights=CROP_CLASS_WEIGHTS,
)
val_base = IHCHierarchicalDataset(
    val_records, input_size=input_size, resize_size=resize_size,
    augment=False, random_crop=False, cache_images=True, four_crop=True,
)
train_dataset = Flat5ClassDataset(train_base, require_labels=True)
print(f'Excluded empty train source patches: {train_dataset.excluded_empty_records}')
val_dataset = Flat5ClassDataset(val_base)

source_counts = count_record_classes(train_records).astype(np.int64)
val_source_counts = count_record_classes(val_records).astype(np.int64)
train_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
val_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
for source_id, class_id in enumerate(SOURCE_TO_FLAT):
    train_counts[class_id] += source_counts[source_id]
    val_counts[class_id] += val_source_counts[source_id]
assert train_counts.sum() == source_counts.sum()
assert val_counts.sum() == val_source_counts.sum()
print('Flat 5-class label counts:')
for class_id, class_name in CLASS_NAMES.items():
    print(f'  [{class_id}] {class_name:24s}: train={train_counts[class_id]:,} | val={val_counts[class_id]:,}')

EPOCH_DATA_REPEATS = 50
batch_size = 16
train_epoch_dataset = torch.utils.data.ConcatDataset(
    [train_dataset] * EPOCH_DATA_REPEATS
)
train_loader = torch.utils.data.DataLoader(
    train_epoch_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=collate_fn, drop_last=False,
    pin_memory=(device.type == 'cuda'), persistent_workers=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    collate_fn=collate_fn, drop_last=False,
    pin_memory=(device.type == 'cuda'), persistent_workers=True,
)

print(f'\nTrain source patches : {len(train_dataset):,} ({len(train_slides)} slides)')
print(f'Samples/epoch        : {len(train_epoch_dataset):,}')
print(f'Train batches        : {len(train_loader):,}')
print(f'Val source/tiles     : {len(val_records)} / {len(val_dataset)}')
cache_mib = (train_dataset.cache_memory_bytes + val_dataset.cache_memory_bytes) / 1024**2
print(f'1024 image cache     : {cache_mib:.1f} MiB')
