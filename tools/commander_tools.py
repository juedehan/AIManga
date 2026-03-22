from agent.features_agent import Features_agent
from langchain_core.tools import tool
from agent.optimize_prompts_agent import Optimize_prompts_agent
from utils.logger_handler import logger
from agent.portrait_agent import Portrait_agent
from typing import Literal, Optional

features_agent = Features_agent()
optimize_prompts_agent = Optimize_prompts_agent()
portrait_agent = Portrait_agent()



def get_features(character_name: str,scene:str) -> str:
    """
    当需要从小说或文本知识库中提取特定人物的详细外貌特征侧写时，必须首先调用此工具。
    【输入】：
            character_name:需要查询的人物姓名或相关身份描述（例如："青灵" 或 "书里的女主角"）。
            scene:需要提取什么场合下的人物特征(例如：“日常”或“打斗”) 。
    【输出】：返回该人物固定格式的文学性特征描述，包含容貌五官、身材体态、衣着装束、气质神韵等。
    """
    logger.info(f"\033[32m[features_agent] Agent调用，提取人物: {character_name}\033[0m")
    desc = ""

    for chunk in features_agent.execute_stream(character_name,scene):
        desc += chunk
    logger.info(f"\033[32m[features_agent] Agent输出\n{desc}\033[0m ")
    return desc


def opt_prompts(literary_description: str,gender: Optional[Literal["男", "女"]] = "未提及") -> str:
    """
    作为画图前的必经步骤！用于将【文学性人物外貌侧写】转化为【豆包视觉大模型专用的高质量中文漫画风格 AI 绘画提示词】。
    【输入】：
        literary_description:必须是 get_features 工具输出的文学性特征描述文本。
        gender:人物的性别，男，女或者未提及
    【输出】：一段逻辑连贯、画面感强烈、且专精于 2D 二次元/漫画风格的纯中文自然语言提示词（纯文本格式）。
    """
    logger.info(f"\033[32m[optimize_prompts_agent] Agent调用\033[0m")
    result = optimize_prompts_agent.execute(literary_description,gender)
    logger.info(f"\033[32m[optimize_prompts_agent] Agent输出\n{result}\033[0m ")
    return result



def get_portrait(optimized_prompt: str, character_name_pinyin: str) -> str:
    """
    用于调用豆包视觉大模型生成人物肖像图，并将图片下载到本地。
    【输入1 optimized_prompt】：必须是 opt_prompts 工具输出的【优化后的绘画提示词】！绝对不能直接输入未经优化的文字。
    【输入2 character_name_pinyin】：当前角色的拼音或英文名（例如：青灵 -> QingLing），不要带任何标点符号或后缀。
    【输出】：生成的图片在本地的绝对保存路径。
    """
    filename = f"{character_name_pinyin}.jpg"
    logger.info(f"\033[32m[portrait_agent] Agent调用，准备生成文件: {filename}\033[0m")

    image_path = portrait_agent.execute(prompt=optimized_prompt, filename=filename)

    logger.info(f"\033[32m[portrait_agent] 最终图片路径: {image_path}\033[0m")
    return image_path