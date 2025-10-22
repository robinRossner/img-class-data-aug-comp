from torch import nn
import torchvision

def build_model(in_channels=3, num_classes=10, name="small_cnn"):
    if name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2()
        if isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, num_classes))
    else:
        raise ValueError(f"Model {name} not recognized.")
    return model
