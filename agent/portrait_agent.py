import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from utils.path_tool import get_project_root,get_abs_path
# 确保加载环境变量
load_dotenv()
class Portrait_agent:
    def __init__(self):
        """
        初始化文生图 Agent，配置客户端和默认的文件保存目录。
        """
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.getenv("ARK_API_KEY")
        )

        # 统一管理图片的保存目录
        image_path = get_abs_path("images")
        self.save_dir = image_path
        # 如果目录不存在，自动创建
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def execute(self, prompt: str, filename: str = "generated_portrait.jpg") -> str:
        """
        接收上游 (Optimize_prompts_agent) 输出的 AI 绘画提示词，
        调用豆包模型生成图片，并下载到本地。

        :param prompt: 优化后的英文或中文绘画提示词
        :param filename: 保存的文件名
        :return: 图片保存的本地绝对路径，如果失败则返回空字符串
        """
        print("正在调用大模型生成图片，请稍候...")
        try:
            # 1. 发起生成请求
            images_response = self.client.images.generate(
                model="doubao-seedream-5-0-260128",
                prompt=prompt,
                size="2K",
                response_format="url",
                extra_body={
                    "watermark": True,
                },

            )

            # 2. 提取 URL
            image_url = images_response.data[0].url
            print(f"图片生成成功！云端临时链接为: {image_url}")

            # 3. 下载图片
            print("正在从云端下载图片到本地...")
            response = requests.get(image_url)

            if response.status_code == 200:
                # 拼接完整的保存路径
                save_path = os.path.join(self.save_dir, filename)

                with open(save_path, "wb") as f:
                    f.write(response.content)

                final_path = os.path.abspath(save_path)
                print(f"下载完成！图片已保存至: {final_path}")
                return final_path
            else:
                print(f"下载失败，服务器返回状态码: {response.status_code}")
                return ""

        except Exception as e:
            print(f" 绘画 Agent 执行过程中发生异常: {e}")
            return ""


if __name__ == '__main__':
    # 测试实例化与执行
    portrait_agent = Portrait_agent()

    test_prompt = "极具视觉张力的校园写实风格，电影级高清画质，8K分辨率，自然光影。画面中央是一位年轻女学生，采用略带仰角的中景镜头突出其修长身形。她皮肤白皙如雪，即便在户外阳光下也泛着细腻光泽，面部轮廓棱角分明，凸显英气；一双大眼炯炯有神，睫毛浓密纤长，小巧嘴唇微抿，神情冷峻。她身穿贴身黑色运动背心与高腰训练短裤，勾勒出因长期锻炼而形成的紧致腰线与饱满臀腿曲线，尤其一双笔直修长的双腿成为视觉焦点。背景为午后阳光洒落的高中操场跑道，远处模糊可见教学楼与树影。光线从侧后方打来，在她肩颈处形成柔和轮廓光，强化疏离感与力量感。整体色调偏冷，但肌肤与布料质感高度真实，细节丰富，呈现“强悍外表下藏有柔软”的微妙氛围。"

    # 执行并获取路径
    result_path = portrait_agent.execute(
        prompt=test_prompt,
        filename="test_student_portrait.jpg"
    )

    print(f"========== 最终输出结果 ==========\n{result_path}")