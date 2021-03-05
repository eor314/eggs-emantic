import cv2
import albumentations as alb


def training_transform(ht=128, wd=128):
    """
    compose the training transforms
    :param ht: height for resize [int]
    :param wd: width for resize [int]
    :return transform:
    """
    t_trans = [
        alb.Resize(height=ht, width=wd),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ShiftScaleRotate(p=0.5,
                             border_mode=cv2.BORDER_REPLICATE,
                            ),
    ]
    
    return alb.Compose(t_trans)


def validation_transform(ht=128, wd=128):
    """
    compose the validation transform (just resize)
    :param ht: height for resize [int]
    :param wd: width for resize [int]
    :return transform:
    """
    v_trans = [
        alb.Resize(height=ht, width=wd),
    ]
    
    return alb.Compose(v_trans)


def to_tensor(x, **kwargs):
    """
    Transform for GPU
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """
    Preprocessing transform
    :param preprocessing: data normalization function (specific for each pretrained neural network)
    :return transform:
    
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(alb.Lambda(image=preprocessing_fn))
    _transform.append(alb.Lambda(image=to_tensor, mask=to_tensor))
    #ToTensor()
    
    return alb.Compose(_transform)