import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class GetModel:
    def __init__(self, num_classes, model_name='fasterrcnn_resnet50_fpn', all_layers=False):
        self.num_classes = num_classes
        print("\n=> using pre-trained model '{}'".format(model_name))
        if model_name == 'fasterrcnn_resnet50_fpn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise ValueError('model is not selected correctly. for Now, just "fasterrcnn_resnet50_fpn" is supported')

        if all_layers==False:
            for param in model.parameters():
                    param.requires_grad = False

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        self.model = model





if __name__ == '__main__':
    print('')