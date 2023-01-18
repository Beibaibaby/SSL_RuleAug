# from .resnet_earlyfunsion import resnet18_earlyfunsion, resnet50_earlyfunsion
# from .resnet_earlyfunsion_moco import resnet18_earlyfunsion_moco, resnet50_earlyfunsion_moco
# from .efnet import EFNet18, EFNet50


from .dcnet import dcnet
from .relnet import relbase
from .mrnet import mrnet

from .mtm import MeanTeacherModel

model_dict = {
    "dcnet": dcnet,
    "relbase": relbase,
    "mrnet": mrnet,
    "dcnet_mtm": lambda x: MeanTeacherModel(base_encoder=dcnet, args=x),
    "relbase_mtm": lambda x: MeanTeacherModel(base_encoder=relbase, args=x),
    "mrnet_mtm": lambda x: MeanTeacherModel(base_encoder=mrnet, args=x)
}


def create_model(args):
    model = None
    model = model_dict[args.arch.lower()](args) 
    return model