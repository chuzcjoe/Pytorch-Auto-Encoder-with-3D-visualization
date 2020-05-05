import os

def Bbox(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()

    return lines[0].split(' ')[1:]

def inference(model, input):
    x = model.encoder.conv1_layer_1(input)
    x = model.encoder.conv1_layer_2(x)
    x = model.encoder.conv2_layer_1(x)
    x = model.encoder.conv2_layer_2(x)
    x = x.view(x.size()[0],-1)
    x = model.encoder.linear1(x)

    return x
