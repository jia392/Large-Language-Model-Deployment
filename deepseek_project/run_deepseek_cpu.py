from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
model_name = "/mnt/data/deepseek-llm-7b-chat"
# 以下部分改变问题
prompt = """
明明明明明白白白喜欢他，可他就是不说。 这句话里，明明和白白谁喜欢谁？
"""

print("正在加载 DeepSeek-LLM-7b-chat 模型...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"       # CPU运行
).eval()

print("正在生成回答，请稍候...")

full_prompt = f"<｜User｜>{prompt}<｜Assistant｜>"
# 编码输入
input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
# 调用标准的generate生成答案
outputs = model.generate(
    input_ids,
    max_new_tokens=512,  # 确保回答完整
    do_sample=False      # 关闭采样，让回答更确定
)

# 解码并截取模型真正回复的内容
response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

print("\n=== 模型回答 ===")
print(response)