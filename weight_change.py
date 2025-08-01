import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger, Qwen2_5_VLModel
from accelerate import Accelerator

# 加载原始 VL 模型和 Qwen3-8B 模型
qwen2_5_vl_7b_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
device = qwen2_5_vl_7b_model.device
qwen3_8b_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map=device,
    torch_dtype=torch.bfloat16
)

# 加载配置
old_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
new_config = AutoConfig.from_pretrained("/home/zhengzhentao/MegaCQA/new_Qwen/new_config.json") # 新 config 的文件夹路径
new_visual_config = new_config.vision_config

# 1. 替换 ViT 到 LLM 的 merger(aligner) 层
new_merger = Qwen2_5_VLPatchMerger(
            dim=new_visual_config.out_hidden_size,
            context_dim=new_visual_config.hidden_size,
            spatial_merge_size=new_visual_config.spatial_merge_size,
        ).to(device).to(torch.bfloat16)
qwen2_5_vl_7b_model.visual.merger = new_merger

# 2. 替换 VL 模型的 LLM 部分
new_llm_model = Qwen2_5_VLModel(new_config).to(device).to(torch.bfloat16)

for name, param in qwen3_8b_model.model.named_parameters():
    if name in new_llm_model.state_dict():
        new_llm_model.state_dict()[name].copy_(param)

qwen2_5_vl_7b_model.model = new_llm_model
qwen2_5_vl_7b_model.lm_head = qwen3_8b_model.lm_head

# 3. 保存修改后的模型
accelerator = Accelerator()
accelerator.save_model(
    model=qwen2_5_vl_7b_model,
    save_directory="/home/zhengzhentao/MegaCQA/new_Qwen",
    max_shard_size="4GB",
    safe_serialization=True
)