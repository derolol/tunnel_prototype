from omegaconf import OmegaConf

import torch

from util.common import instantiate_from_config, load_state_dict


def main() -> None:
    config_path = "/home/lib/generate_seg/config/model/segformerb0.yaml"
    model_config = OmegaConf.load(config_path)

    print(f"Start testing model {model_config.target}")

    model = instantiate_from_config(model_config)

    input_tensor = torch.rand(size=(1, 3, 512, 512))
    print(f"Test input shape: {input_tensor.shape}")

    output = model(input_tensor)
    if isinstance(output, list) or isinstance(output, tuple):
        for i in len(output):
            if isinstance(output[i], torch.Tensor):
                print(f"Test output{i} shape: ", output[i].shape)
            else:
                print(f"Miss output{i} type: {type(output[i])}")
    elif isinstance(output, torch.Tensor):
        print(f"Test output shape: ", output.shape)
    else:
        print(f"Miss output type: {type(output)}")
    
if __name__ == "__main__":
    main()
