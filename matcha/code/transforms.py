import cv2
import albumentations as albu


def blur_transforms(p=0.5, blur_limit=5):
    return albu.OneOf(
        [
            albu.GaussianBlur(always_apply=True, sigma_limit=0.5),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(50, 150), always_apply=True),
            albu.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.2, always_apply=True
            ),
            albu.ChannelShuffle(always_apply=True),
            albu.ToGray(always_apply=True),
        ],
        p=p,
    )


def get_transfos(strength=1):

    if strength == 0:
        augs = []
    elif strength == 1:
        augs = [
            color_transforms(p=0.25),
        ]
    elif strength == 2:
        augs = [
            color_transforms(p=0.5),
            albu.ImageCompression(quality_lower=50, quality_upper=100, p=0.25),
            albu.RandomScale(scale_limit=(0.5, 1.25), p=0.25),
        ]
    elif strength == 3:
        augs = [
            color_transforms(p=0.75),
            albu.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
            albu.RandomScale(scale_limit=(0.5, 1.25), p=0.5),
        ]

    return albu.Compose(augs)