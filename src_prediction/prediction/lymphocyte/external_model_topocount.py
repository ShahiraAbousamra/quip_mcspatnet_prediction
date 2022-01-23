import numpy as np
import sys
sys.path.append("..");
sys.path.append(".");
sys.path.append("../..");
sys.path.append("...");
from topocount_src.run_external_input import load_model;


def load_external_model(model_path):
    # Load your model here
    model = load_model(model_path)
    return model

def pred_by_external_model(model, inputs):
    pred = model.predict(inputs)

    return pred;

if __name__ == "__main__":
    

    model_path = "/gpfs/projects/KurcGroup/sabousamra/cc/TIL/u24_lymphocyte/prediction/topocount_models/topocount_model.pth";

    print(model_path)
    model = load_external_model(model_path);
    print('load_external_model called')
    inputs = np.random.rand(10, 3, 100, 100);
    print('inputs created')
    pred = pred_by_external_model(model, inputs)
    print('after predict')
    print(pred);
