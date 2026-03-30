from typing import Literal, Optional
import time

from agent.features_agent import Features_agent
from agent.optimize_prompts_agent import Optimize_prompts_agent
from agent.portrait_adjustment_agent import PortraitAdjustmentAgent
from agent.portrait_agent import Portrait_agent
from tools.agent_tools import (
    load_character_portrait_memory,
    save_character_portrait_memory,
    update_character_portrait_memory,
)
from utils.logger_handler import logger

features_agent = Features_agent()
optimize_prompts_agent = Optimize_prompts_agent()
portrait_agent = Portrait_agent()


def get_features(character_name: str, scene: str) -> str:
    """
    当需要首次为某个角色生成肖像，或者用户明确要求重新从小说中提取该角色在特定场景下的人物特征时，调用此工具。
    【输入】：
        character_name: 需要查询的人物姓名或相关身份描述。
        scene: 需要提取什么场合下的人物特征，例如“日常”或“打斗”。
    【输出】：
        返回该人物固定格式的文学性特征描述，包含容貌五官、身材体态、衣着装束、气质神韵等。
    """
    logger.info(f"\033[32m[features_agent] Agent调用，提取人物: {character_name}\033[0m")
    desc = ""

    for chunk in features_agent.execute_stream(character_name, scene):
        desc += chunk
    logger.info(f"\033[32m[features_agent] Agent输出\n{desc}\033[0m ")
    return desc


def opt_prompts(literary_description: str, gender: Optional[Literal["男", "女"]] = "未提及") -> str:
    """
    当需要为首次生成的新肖像准备绘画提示词时，调用此工具。
    【输入】：
        literary_description: 必须是 get_features 工具输出的文学性特征描述文本。
        gender: 人物的性别，男、女或者未提及。
    【输出】：
        返回一段逻辑连贯、画面感强烈、专用于 2D 漫画风格出图的最终绘画提示词。
    """
    logger.info(f"\033[32m[optimize_prompts_agent] Agent调用\033[0m")
    result = optimize_prompts_agent.execute(literary_description, gender)
    logger.info(f"\033[32m[optimize_prompts_agent] Agent输出\n{result}\033[0m ")
    return result


def get_portrait(optimized_prompt: str, character_name_pinyin: str) -> str:
    """
    用于调用文生图模型生成人物肖像图，并将图片下载到本地。
    【输入1 optimized_prompt】：必须是 opt_prompts 工具输出的优化后绘画提示词。
    【输入2 character_name_pinyin】：当前角色的拼音或英文名，例如：青灵 -> QingLing。
    【输出】：生成的图片在本地的绝对保存路径。
    """
    filename = f"{character_name_pinyin}.jpg"
    logger.info(f"\033[32m[portrait_agent] Agent调用，准备生成文件: {filename}\033[0m")

    image_path = portrait_agent.execute(prompt=optimized_prompt, filename=filename)

    logger.info(f"\033[32m[portrait_agent] 最终图片路径: {image_path}\033[0m")
    return image_path


class CommanderToolbox:
    def __init__(self):
        self.features_agent = Features_agent()
        self.optimize_prompts_agent = Optimize_prompts_agent()
        self.portrait_agent = Portrait_agent()
        self.portrait_adjustment_agent = PortraitAdjustmentAgent()
        self.current_session_id: str | None = None
        self.current_user_query: str = ""

    def set_request_context(self, session_id: str, user_query: str) -> None:
        self.current_session_id = session_id
        self.current_user_query = user_query

    def _require_session_id(self) -> str:
        if not self.current_session_id:
            raise RuntimeError("CommanderToolbox 尚未设置 session_id")
        return self.current_session_id

    def _build_filename(self, character_name_pinyin: str, suffix: str = "") -> str:
        timestamp = int(time.time())
        if suffix:
            return f"{character_name_pinyin}_{suffix}_{timestamp}.jpg"
        return f"{character_name_pinyin}_{timestamp}.jpg"

    def get_features(self, character_name: str, scene: str) -> str:
        """
        适用于首次生成某个角色的肖像，或用户明确要求重新从小说文本中提取该角色在特定场景下的人物特征时使用。
        【一定要在以下场景调用】：
        1. 用户第一次要求画某个角色，没有任何历史肖像可以沿用。
        2. 用户明确要求重新参考小说原文、重新做人设、重新提取外貌特征。
        3. 用户更换了角色，不再沿用之前已经画过的角色。
        【不要在以下场景调用】：
        1. 用户只是对已经生成过的同一角色肖像不满意，想修改发色、服饰、动作、表情、神态、镜头、构图、色调、背景等局部内容。
        2. 用户想在已有最终提示词基础上微调，而不是重新检索小说。
        【输入】：
            character_name: 需要查询的人物姓名或身份描述。
            scene: 需要提取什么场合下的人物特征，例如“日常”“打斗”“校园”。
        【输出】：
            该角色的人物特征侧写文本。
        """
        logger.info(f"\033[32m[features_agent] Agent调用，提取人物: {character_name}\033[0m")
        desc = ""
        for chunk in self.features_agent.execute_stream(character_name, scene):
            desc += chunk
        logger.info(f"\033[32m[features_agent] Agent输出\n{desc}\033[0m ")
        return desc

    def opt_prompts(self, literary_description: str, gender: Optional[Literal["男", "女"]] = "未提及") -> str:
        """
        适用于首次生成角色肖像时，把文学性人物描述转换为最终绘画提示词。
        【一定要在以下场景调用】：
        1. 已经通过 get_features 拿到了角色特征描述，准备首次出图。
        2. 用户明确要求重新生成该角色的完整提示词，而不是在已有提示词上局部修改。
        【不要在以下场景调用】：
        1. 用户只是对同一角色的已有肖像做局部修改。
        2. 已经有该角色的历史最终提示词，可以直接在原提示词上做增量调整。
        【输入】：
            literary_description: 必须是 get_features 输出的人物特征描述。
            gender: 人物性别。
        【输出】：
            一段完整的、可直接用于文生图的最终绘画提示词。
        """
        logger.info(f"\033[32m[optimize_prompts_agent] Agent调用\033[0m")
        result = self.optimize_prompts_agent.execute(literary_description, gender)
        logger.info(f"\033[32m[optimize_prompts_agent] Agent输出\n{result}\033[0m ")
        return result

    def get_portrait(
        self,
        optimized_prompt: str,
        character_name: str,
        character_name_pinyin: str,
        scene: str = "未提及",
        gender: Optional[Literal["男", "女"]] = "未提及",
    ) -> str:
        """
        适用于首次生成角色肖像，或用户明确要求重新完整绘制某个角色时使用。
        【前置条件】：
        1. optimized_prompt 必须已经是完整的最终绘画提示词。
        2. 首次生成场景下，通常应当先经过 get_features 和 opt_prompts。
        【工具行为】：
        1. 调用文生图模型出图。
        2. 在当前会话下，把该角色的最终提示词和出图结果写入 portrait memory，供后续局部修改时直接复用。
        【输入】：
            optimized_prompt: 最终绘画提示词。
            character_name: 角色名，用于写入会话级 portrait memory。
            character_name_pinyin: 角色拼音或英文名，用于生成文件名。
            scene: 本次绘制对应的场景说明。
            gender: 角色性别。
        【输出】：
            生成图片的本地绝对路径；若失败则返回空字符串。
        """
        filename = self._build_filename(character_name_pinyin)
        logger.info(f"\033[32m[portrait_agent] Agent调用，准备生成文件: {filename}\033[0m")
        image_path = self.portrait_agent.execute(prompt=optimized_prompt, filename=filename)

        if image_path:
            save_character_portrait_memory(
                session_id=self._require_session_id(),
                character_name=character_name,
                character_name_pinyin=character_name_pinyin,
                latest_final_prompt=optimized_prompt,
                latest_image_path=image_path,
                latest_scene=scene,
                latest_gender=gender,
                last_user_request=self.current_user_query,
            )

        logger.info(f"\033[32m[portrait_agent] 最终图片路径: {image_path}\033[0m")
        return image_path

    def adjust_existing_portrait(
        self,
        modification_request: str,
        character_name: Optional[str] = None,
        character_name_pinyin: Optional[str] = None,
    ) -> str:
        """
        适用于用户对当前会话中已经画过的角色肖像不满意，希望在现有最终提示词基础上做局部调整时使用。
        【最适合处理的需求】：
        - 修改发色、发型、服饰、动作、姿势、表情、神态、镜头、构图、色调、氛围、背景等
        - “把刚才那张改一下”“在上一版基础上调整”“不要重做人设，只改局部细节”
        【不会做的事】：
        - 不重新检索小说知识库
        - 不重新提取人物特征
        - 不重新走提示词优化 Agent
        【前置条件】：
        - 当前会话下必须已经存在该角色的历史肖像记录
        - 如果未显式提供 character_name，则默认尝试使用当前会话最近一次成功生成或调整过的角色
        【miss 行为】：
        - 若找不到该角色的历史肖像记录，直接返回失败原因，不自动回退到冷启动流程
        【输入】：
            modification_request: 用户本轮要修改的具体要求
            character_name: 可选，目标角色名；省略时回退到当前会话最近角色
            character_name_pinyin: 可选，若不提供则优先使用 memory 中已有的拼音
        【输出】：
            成功时返回新的图片绝对路径；失败时返回明确失败原因文本。
        """
        session_id = self._require_session_id()
        memory_record = load_character_portrait_memory(session_id=session_id, character_name=character_name)
        if memory_record is None:
            if character_name:
                return f"未找到当前会话下角色“{character_name}”的历史肖像记录，请先生成该角色的初始肖像。"
            return "未找到当前会话下最近一次角色的历史肖像记录，请先生成初始肖像。"

        resolved_character_name = memory_record["character_name"]
        resolved_character_pinyin = (
            character_name_pinyin
            or memory_record.get("character_name_pinyin")
            or resolved_character_name
        )
        updated_prompt = self.portrait_adjustment_agent.execute(
            base_prompt=memory_record["latest_final_prompt"],
            modification_request=modification_request,
            character_name=resolved_character_name,
        )
        if not updated_prompt:
            return "角色肖像调整失败，未生成新的绘画提示词。"

        filename = self._build_filename(resolved_character_pinyin, suffix="adjust")
        image_path = self.portrait_agent.execute(prompt=updated_prompt, filename=filename)
        if not image_path:
            return "角色肖像调整失败，图片生成未成功。"

        update_character_portrait_memory(
            session_id=session_id,
            character_name=resolved_character_name,
            character_name_pinyin=resolved_character_pinyin,
            latest_final_prompt=updated_prompt,
            latest_image_path=image_path,
            modification_request=modification_request,
            latest_scene=memory_record.get("latest_scene"),
            latest_gender=memory_record.get("latest_gender"),
            last_user_request=self.current_user_query,
        )
        logger.info(
            f"\033[32m[portrait_adjustment_agent] 完成角色调整: {resolved_character_name}, 图片路径: {image_path}\033[0m"
        )
        return image_path
