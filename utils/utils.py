from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import transforms

# The training images are local, trusted data. Allow Pillow to decode the one
# very large image and recover JPEGs missing a small trailing byte segment.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform = None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not self.files:
            raise ValueError(f'No JPG or PNG images found in: {root}')
        self._skipped_files = set()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # A single corrupt or partially downloaded image should not stop a
        # multi-hour training run. Try the next image instead and make the
        # skipped filename visible in the console.
        for offset in range(len(self.files)):
            file_idx = (idx + offset) % len(self.files)
            image_path = os.path.join(self.root, self.files[file_idx])
            try:
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert('RGB')
                    image.load()  # Force decoding while the file is open.

                if self.transform:
                    image = self.transform(image)
                return image
            except (OSError, UnidentifiedImageError) as exc:
                if image_path not in self._skipped_files:
                    self._skipped_files.add(image_path)
                    print(
                        f'\n[Skipped image; training continues] {image_path}\n'
                        f'Reason: {exc}\n',
                        flush=True,
                    )

        raise RuntimeError(f'No readable images found in: {self.root}')


def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        

def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std
