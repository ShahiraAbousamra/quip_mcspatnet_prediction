import numpy as np
import sys
sys.path.append("..");
sys.path.append(".");
sys.path.append("../..");
sys.path.append("...");
from mcspatnet_src.run_external_input import load_model;


def load_external_model(model_path):
    # Load your model here
    model = load_model(model_path)
    return model

def pred_by_external_model(model, inputs):
    pred = model.predict(inputs)

    return pred;

if __name__ == "__main__":
    

    model_path = "/home/sabousamra/quip_mcspatnet_prediction/src_prediction/prediction/mcspatnet_models/mcspat_brca-m2c.pth";

    print(model_path)
    model = load_external_model(model_path);
    print('load_external_model called')
    inputs = np.random.rand(10, 3, 100, 100);
    print('inputs created')
    pred = pred_by_external_model(model, inputs)
    print('after predict')
    print(pred);
