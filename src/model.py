from torch import nn
import torchvision


class SmallCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__() # Behave like a nn.Module with same methods etc
        self.ConvStack = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )


    def forward(self, x):
        x = self.ConvStack(x)
        x = self.classifier(x)
        return x


def build_model(in_channels=3, num_classes=10, name="small_cnn"):
    if name == "small_cnn":
        model = SmallCNN(in_channels=in_channels, num_classes=num_classes)
    elif name == "resnet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=False)
        if isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, num_classes))
    else:
        raise ValueError(f"Model {name} not recognized.")
    return model
