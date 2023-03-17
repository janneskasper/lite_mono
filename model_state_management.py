import os
import json
import torch


def save_model_config(options, log_path):
    """Save options to disk so we know what we ran this experiment with
    """
    models_dir = os.path.join(log_path, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    with open(os.path.join(models_dir, 'model_config.json'), 'w') as f:
        json.dump(options.__dict__.copy(), f, indent=2)

def save_model(log_path, name, models, options, optimizier):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_path, "models", name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            to_save['height'] = options.height
            to_save['width'] = options.width
            to_save['use_stereo'] = options.use_stereo
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(optimizier.state_dict(), save_path)

def load_model(weights_folder, models_to_load, models, optimizer):
    """Load model(s) from disk
    """
    load_weights_folder = os.path.expanduser(weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))

    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        model_dict = models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[n].load_state_dict(model_dict)

    # loading adam state
    optimizer_load_path = os.path.join(load_weights_folder, "adam.pth")
    if os.path.isfile(optimizer_load_path):
        print("Loading Adam weights")
        optimizer_dict = torch.load(optimizer_load_path)
        optimizer.load_state_dict(optimizer_dict)
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")

