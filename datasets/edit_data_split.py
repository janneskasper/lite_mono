import os
import random
import argparse

def load_split_file(filepath):
    assert os.path.isfile(filepath), f"Load split file: path {filepath} is not pointing to a file!"

    scenarios = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

        for line in lines:
            scenario, frame_id, stereo_side = line.split()
            if scenario in scenarios.keys():
                scenarios[scenario].append((frame_id, stereo_side))
            else:
                scenarios[scenario] = [(frame_id, stereo_side)]
    
    return scenarios

def change_split_file(filepath, out_path, remove_every_x_scenario=None, remove_x=None, shuffle=True):
    assert remove_every_x_scenario is not None or remove_x is not None , f"Specify either number of scenes to remove or number of every x scene to remove"

    scenarios = load_split_file(filepath)
    
    keys = list(scenarios.keys())
    if shuffle:
        random.shuffle(keys)
    
    if remove_every_x_scenario is not None:
        shuffled_keys = [item for index, item in enumerate(keys) if index % remove_every_x_scenario != 0]
    elif remove_x is not None:
        shuffled_keys = keys[remove_x:]

    lines = []
    for key in shuffled_keys:
        for frame in scenarios[key]:
            lines.append(f"{key} {frame[0]} {frame[1]}\n")

    with open(out_path, "w") as f:
        f.writelines(lines)

def split_editor(filepath):
    scenarios = load_split_file(filepath)

    keys = list(scenarios.keys())

    while True:
        command_list = f"-----------------------------------------\n"\
                        "Command list:\n"\
                        "- q: Quit\n" \
                        "- s: Save\n" \
                        "- l: List all scenes\n" \
                        f"- 0-{len(keys)}: Number to delete\n" \
                        "-----------------------------------------\n"
        print(command_list)
        key_in = str(input("Enter command:\n")).lower()
        
        if key_in == "q":
            break
        elif key_in == "l":
            scenes = ""
            for i,k in enumerate(keys):
                scenes += f"{i}: {k}, Num. Images: {len(scenarios[k])}\n"
            print(scenes)
        elif key_in == "s":
            out_path = str(input("Enter output path:\n")).lower()

            lines = []
            for key in keys:
                for frame in scenarios[key]:
                    lines.append(f"{key} {frame[0]} {frame[1]}\n")
            with open(out_path, "w") as f:
                f.writelines(lines)
            print(f"==> Saved file to {out_path} <==\n")
        else:
            try:
                index = int(key_in)
            except:
                print("Please enter a valid command!\n")
                continue
            index = min(len(scenarios)-1, max(index,0))
            keys.pop(index)
            print(f"==> Removed scene {index} <==\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Speficies the input split file path")
    parser.add_argument("-o", "--output", type=str, help="Speficies the output split file path for non interactive")
    parser.add_argument("-d", "--delete_cnt", type=int, default=1, help="Speficies the number of random scenes to delete")
    parser.add_argument("--interactive", action="store_true", help="Starts interactive split removing")
    args = parser.parse_args()

    assert args.input is not None, f"Please specify input and output file path"
    assert os.path.isfile(args.input), "Input path is not correctly specified"
    
    if not args.interactive and args.output is None:
        print("When not using interactive mode specify an output path")
        exit(0)

    if args.interactive:
        split_editor(args.input)
    else:
        change_split_file(args.input, args.output, remove_x=args.delete_cnt)