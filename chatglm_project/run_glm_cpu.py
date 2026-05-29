from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径为智谱模型路径
model_name = "/mnt/data/chatglm3-6b" 
# 以下部分改变问题
prompt = (
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 "
    "领导：你这就不够意思了。 小明：小意思，小意思。 "
    "领导：你这人真有意思。 小明：其实也没有别的意思。 "
    "领导：那我就不好意思了。 小明：是我不好意思。 "
    "请问：以上“意思”分别是什么意思。"
)

print("正在加载 ChatGLM3-6B 模型...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"  # 显式指定在CPU上运行
).eval()

print("正在生成回答，请稍候...")
# 调用ChatGLM3官方标准的对话接口
response, history = model.chat(tokenizer, prompt, history=[])

print("\n=== 模型回答 ===")
print(response)