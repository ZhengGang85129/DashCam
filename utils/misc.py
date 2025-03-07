import torch.nn as nn
import logging
import argparse 

def parse_args(parser_name: str = 'DEFAULT')-> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f'{parser_name} script with batch size argument')

    parser.add_argument('--model_dir', type=str, default='model',
                        help='directory to save models (default: ./model)')
    parser.add_argument('--debug', action = "store_true", help = 'Activate to turn on the debug mode')

    args = parser.parse_args()
    return args

def print_trainable_parameters(model: nn.Module, logger:logging.Logger) -> None:
    """Print only the trainable parts of the model architecture"""
    
    for name, module in model.named_children():
        # Check if any parameters in this module are trainable
        has_trainable = any(p.requires_grad for p in module.parameters())
        
        if has_trainable:
            logger.info(f"\n{name}:")
            # Check if it's a container module (like Sequential)
            if isinstance(module, nn.Sequential) or len(list(module.children())) > 0:
                # Print each trainable submodule
                for sub_name, sub_module in module.named_children():
                    sub_trainable = any(p.requires_grad for p in sub_module.parameters())
                    if sub_trainable:
                        logger.info(f"  {sub_name}: {sub_module}")
            else:
                # Print the module directly
                print(f"  {module}")
