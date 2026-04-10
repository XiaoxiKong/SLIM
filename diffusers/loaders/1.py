def save_lora_parameters(model, filepath):
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params[name] = param.detach().cpu()

    torch.save(lora_params, filepath)
    print(f"LoRA parameters saved to {filepath}")


# 加载 LoRA 参数
def load_lora_parameters(model, filepath):
    lora_params = torch.load(filepath)
    model_state = model.state_dict()

    for name, param in lora_params.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Parameter {name} not found in model. Skipping.")

    model.load_state_dict(model_state)
    print(f"LoRA parameters loaded from {filepath}")