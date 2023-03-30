from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import time
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from prettytable import PrettyTable

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')
    parser.add_argument("--test_files", type=str, help="Path to a single image file or directory")
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images')

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use', required=True
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono",
                        choices=[
                            "lite-mono",
                            "lite-mono-small",
                            "lite-mono-tiny"])

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    
    parser.add_argument("--model_summary", action="store_true", help="Prints a model summary")

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert os.path.isdir(args.load_weights_folder), "--load_weights_folder has to point the directory with the saved weights"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.LiteMono(model=args.model,
                                    height=feed_height,
                                    width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path) and not args.test:
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isfile(args.image_path) and args.test:
        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        # reading images from .txt file
        paths = []
        with open(args.image_path) as f:
            filenames = f.readlines()
            for i in range(len(filenames)):
                filename = filenames[i]
                line = filename.split()
                folder = line[0]
                if len(line) == 3:
                    frame_index = int(line[1])
                    side = line[2]

                f_str = "{:010d}{}".format(frame_index, '.jpg')
                image_path = os.path.join(
                    'kitti_data',
                    folder,
                    "image_0{}/data".format(side_map[side]),
                    f_str)
                paths.append(image_path)

    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # output_name = os.path.splitext(image_path)[0].split('/')[-1]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))


    print('-> Done!')

def test_lite_mono(args):
    """Function to predict for a single image or folder of images
    """
    assert os.path.isdir(args.load_weights_folder), "--load_weights_folder has to point the directory with the saved weights"
    assert os.path.isfile(args.test_files) is not None, "--test_files has to point to a single image file or a directory with files"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.LiteMono(model=args.model,
                                    height=feed_height,
                                    width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    image_paths = []
    if os.path.isdir(args.test_files):
        image_paths = glob.glob(os.path.join(args.test_files, '*.{}'.format(args.ext)))
    elif os.path.isfile(args.test_files):
        image_paths = [args.test_files]
    if len(image_paths) < 1:
        print(f"No images found with ending \"{args.ext}\" in \"{args.test_files}\"!")
        return
    with torch.no_grad():
        # Load image and preprocess
        i = 1
        for img in image_paths:
            input_image = pil.open(img).convert('RGB')
            original_width, original_height = input_image.size
            input_image_tensor = transforms.ToTensor()(input_image.resize((feed_width, feed_height), pil.LANCZOS)).unsqueeze(0)

            # PREDICTION
            # do it a few times for real performance -> first inference quite slow
            t = time.time()
            input_image_tensor = input_image_tensor.to(device)
            features = encoder(input_image_tensor)
            outputs = depth_decoder(features)
            print(f"{i}: Inference time: {(time.time() - t) * 1000} ms")
            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = np.array((mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8))

            cv2.imshow("Input", np.array(input_image))
            cv2.imshow("Colormap", cv2.cvtColor(colormapped_im, cv2.COLOR_BGR2RGB))
            if len(image_paths) > 1:
                cv2.waitKey(delay=50)
            else:
                cv2.waitKey(0)
            i+=1

def print_model_summary(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']
    encoder = networks.LiteMono(model=args.model,
                                    height=feed_height,
                                    width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)

    encoder.cuda()
    
    print(encoder)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in encoder.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    
    del depth_decoder
    del encoder

if __name__ == '__main__':
    args = parse_args()
    assert args.test_files is not None or args.image_path is not None or args.model_summary is not None, "You must specify either --image_path or --test_files"
    assert args.load_weights_folder is not None, "You must specify the --load_weights_folder parameter"

    if args.model_summary:
        print_model_summary(args)

    if args.test_files:
        test_lite_mono(args)
    elif args.image_path:
        test_simple(args)
