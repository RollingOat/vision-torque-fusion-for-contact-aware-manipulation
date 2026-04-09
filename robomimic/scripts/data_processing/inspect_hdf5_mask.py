import h5py
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Inspect 'mask' group of an HDF5 file.")
    parser.add_argument("file", help="Path to the HDF5 file")
    args = parser.parse_args()

    try:
        with h5py.File(args.file, "r") as f:
            print(f"Inspecting file: {args.file}")
            
            

            mask_group = f["mask"]
            print(f"Found 'mask' group with keys: {list(mask_group.keys())}")

            for key in mask_group.keys():
                print("-" * 40)
                print(f"Key: {key}")
                data = mask_group[key][()]
                print(f"Shape: {data.shape}")
                print(f"Type: {data.dtype}")
                
                # Print content (limited)
                print("Data:")
                if data.size == 0:
                    print("  <Empty>")
                else:
                    # Convert bytes to utf-8 if necessary for display
                    display_data = data
                    if data.dtype.kind == 'S' or data.dtype.kind == 'O': 
                        try:
                            display_data = [d.decode('utf-8') if isinstance(d, bytes) else d for d in data]
                        except:
                            pass
                    
                    print(display_data)

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
