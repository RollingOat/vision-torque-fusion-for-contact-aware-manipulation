import h5py
import argparse
import os
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Combine multiple Robomimic HDF5 datasets into one.")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of input HDF5 files to combine")
    parser.add_argument("--out", type=str, required=True, help="Output HDF5 file path")
    parser.add_argument("--train_percent", type=float, default=0.9, help="Percentage of demos to use for training (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for train/valid split")
    
    args = parser.parse_args()
    
    if os.path.exists(args.out):
        print(f"Warning: Output file {args.out} already exists. It will be overwritten.")
    
    # Open output file
    # We use 'w' to create a new file (overwrite if exists)
    with h5py.File(args.out, "w") as f_out:
        g_data_out = f_out.create_group("data")
        
        total_samples = 0
        demo_count = 0
        output_demo_names = []
        env_args = None
        
        for input_path in args.inputs:
            if not os.path.exists(input_path):
                print(f"Warning: Input file {input_path} does not exist. Skipping.")
                continue
                
            print(f"Processing {input_path}...")
            
            try:
                with h5py.File(input_path, "r") as f_in:
                    if "data" not in f_in:
                        print(f"Warning: 'data' group not found in {input_path}. Skipping.")
                        continue
                    
                    g_data_in = f_in["data"]
                    
                    # Get env_args from the first valid file that has it
                    if env_args is None and "env_args" in g_data_in.attrs:
                        env_args = g_data_in.attrs["env_args"]
                    
                    # Sort demos to maintain order
                    # Heuristic: split by '_' and convert headers to int if possible
                    # E.g. demo_0, demo_1 ...
                    demo_keys = list(g_data_in.keys())
                    
                    def demo_sort_key(x):
                        parts = x.split('_')
                        if len(parts) > 1 and parts[1].isdigit():
                            return int(parts[1])
                        return x
                        
                    demos = sorted(demo_keys, key=demo_sort_key)
                    
                    for demo_key in demos:
                        source_demo = g_data_in[demo_key]
                        
                        # New demo name
                        new_demo_name = f"demo_{demo_count}"
                        
                        output_demo_names.append(new_demo_name)
                        
                        # Copy the group using the destination file object
                        # We copy the source object to the destination path inside output file
                        dest_path = f"data/{new_demo_name}"
                        f_out.copy(source_demo, dest_path)
                        
                        # Get num_samples
                        n_samples = 0
                        if "num_samples" in source_demo.attrs:
                            n_samples = source_demo.attrs["num_samples"]
                        elif "actions" in source_demo:
                            n_samples = source_demo["actions"].shape[0]
                        elif "obs" in source_demo:
                            # Try to find a key in obs
                            obs_group = source_demo["obs"]
                            if len(obs_group.keys()) > 0:
                                first_key = list(obs_group.keys())[0]
                                n_samples = obs_group[first_key].shape[0]
                        
                        # Verify num_samples attribute exists in copied demo
                        # because we just copied the group, attributes should be there
                        # But if it wasn't there in source, we might want to add it
                        if "num_samples" not in f_out[dest_path].attrs:
                            f_out[dest_path].attrs["num_samples"] = n_samples

                        total_samples += n_samples
                        demo_count += 1
                        
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                import traceback
                traceback.print_exc()

        # Write global attributes
        g_data_out.attrs["total"] = total_samples
        
        if env_args is None:
             # Default env_args if none found
             default_env_args = {"env_name": "Unknown", "env_kwargs": {}}
             g_data_out.attrs["env_args"] = json.dumps(default_env_args)
        else:
            g_data_out.attrs["env_args"] = env_args

        mask_grp = f_out.create_group("mask")
        # Clamp train-percent to [0,1]
        p = float(args.train_percent)
        p = max(0.0, min(1.0, p))

        # IMPORTANT: ensure N matches the number of demos we actually processed
        # The 'output_demo_names' list has all the keys like "demo_0", "demo_1", etc.
        N = len(output_demo_names)
        n_train = int(np.floor(p * N))

        # Randomly assign demos to train/valid. Use seed if provided for reproducibility.
        if N > 0:
            if args.seed is not None:
                rng = np.random.RandomState(args.seed)
                perm = rng.permutation(N)
            else:
                perm = np.random.permutation(N)

            train_idx = perm[:n_train]
            valid_idx = perm[n_train:]

            # Index into the list of actual demo names
            train_names = [output_demo_names[i] for i in train_idx]
            valid_names = [output_demo_names[i] for i in valid_idx]
        else:
            train_names = []
            valid_names = []
            
        print(f"Assigned {len(train_names)} demos to train and {len(valid_names)} demos to validation.")
        # Only print first few to avoid spamming if list is huge
        if len(train_names) > 0:
             print(f"Train demos (first 5): {train_names[:5]} ...")
        if len(valid_names) > 0:
             print(f"Validation demos (first 5): {valid_names[:5]} ...")

        dt = h5py.string_dtype(encoding='utf-8')
        # Write train/valid lists into the top-level mask group
        # Robomimic expects these to be datasets of strings
        mask_grp.create_dataset("train", data=np.array(train_names, dtype=dt))
        mask_grp.create_dataset("valid", data=np.array(valid_names, dtype=dt))
            
        print(f"\nSuccessfully combined datasets.")
        print(f"Total demos: {demo_count}")
        print(f"Total samples: {total_samples}")
        print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
