from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/Qwen-7B-Chat" # 本地路径
# 以下部分改变问题
prompt = "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。"

print("正在加载 Qwen-7B-Chat 模型...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"       # 指定在CPU上运行，防止环境报错
).eval()

# 使用Qwen官方标准的model.chat接口
print("正在生成回答，请稍候...")
response, history = model.chat(tokenizer, prompt, history=None)

print("\n=== 模型回答 ===")
print(response)