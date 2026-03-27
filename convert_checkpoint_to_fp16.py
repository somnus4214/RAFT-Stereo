import torch
import os

def convert_to_fp16(input_path, output_path):
    print(f"Reading original checkpoint from: {input_path}")
    
    # Load the checkpoint
    state_dict = torch.load(input_path, map_location='cpu')
    
    # Create a new state dict to hold the converted weights
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Check if the value is a tensor and if it's a floating point type
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            new_state_dict[key] = value.half()  # Convert to float16
        else:
            new_state_dict[key] = value  # Keep non-floating-point parameters unchanged

    print(f"Saving FP16 checkpoint to: {output_path}")
    torch.save(new_state_dict, output_path)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    input_file = "models/raftstereo-middlebury.pth"
    output_file = "models/raftstereo-middlebury-fp16.pth"
    
    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Required checkpoint {input_file} does not exist.")
    else:
        convert_to_fp16(input_file, output_file)
