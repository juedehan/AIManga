# AIManga
将小说变成漫画的Agent
使用时需要自行在项目根目录下新建`.env`文件存放相关大模型的API-key，示例如下
```
OPENAI_API_KEY=xxxxxxx
DASHSCOPE_API_KEY=xxxxxxx

ARK_API_KEY=xxxxxxx
```
前两个API-key是相同的，都是百炼平台的大模型，用于LLM,文本嵌入以及rerank，第三个是文生图模型的apikey

默认使用的大模型如下
```aiignore
# LLM大模型
chat_model_name: qwen3-max
# 文本嵌入向量模型
embedding_model_name: text-embedding-v4
# rerank模型
rerank_model_name: qwen3-rerank

# 文生图大模型名称
image_model_name: doubao-seedream-5-0-260128
```
因为相关大模型API调用格式已经写好，推荐按照默认大模型来练习，
搞明白后可以在`model/factory.py`中自行更换模型

其中LLM,embedding与rerank模型在百炼平台,新人注册有免费token额度
>https://bailian.console.aliyun.com/

大学生还可以进行认证，在高校认证后还能有300人民币额度 
>https://university.aliyun.com/?spm=5176.30260724.J_4NWEMkQ5nDwOgLi8EJmHs.23.15c7db575OivtJ

目前只做了给人物画肖像的功能，后续还会增加场景分镜等