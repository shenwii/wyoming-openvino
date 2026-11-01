import openvino as ov
import numpy as np
import asyncio
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.wakeword import Detect, Detection
from wyoming.audio import AudioChunk

# --- OpenVINO 初始化 ---
core = ov.Core()
core.set_property("GPU", {"PERFORMANCE_HINT": "LATENCY"})
print("正在将唤醒模型加载到 GPU...")
# OpenVINO 2023+ 原生支持读取 tflite 或 onnx
model = core.read_model("my_wakeword_model.tflite")
compiled_model = core.compile_model(model, "GPU")

class OV_WakeWord_Handler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 维护一个滑动窗口缓存 (比如保留最近的音频特征)
        self.audio_buffer = np.zeros(EXPECTED_INPUT_SIZE, dtype=np.float32)

    async def handle_event(self, event):
        if Detect.is_type(event.type):
            # HA 请求开始唤醒词检测
            self.audio_buffer.fill(0)

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            # 1. 解析二进制音频切片
            audio_array = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32)

            # 2. 更新滑动窗口，并在需要时提取梅尔频谱特征 (Melspectrogram)
            # update_sliding_window(audio_array)
            # features = extract_features(self.audio_buffer)

            # 3. 交给 OpenVINO GPU 推理
            # result 是一个概率值列表
            result = compiled_model([features])[0]
            probability = result[0][0]

            # 4. 判断阈值，触发唤醒
            if probability > 0.65:
                print(f"🔥 唤醒成功! 概率: {probability}")
                # 告诉 HA 听到了唤醒词
                await self.write_event(Detection(name="my_model", timestamp=chunk.timestamp).event())
                # 唤醒后清空缓存，防止连续重复触发
                self.audio_buffer.fill(0)

        return True

async def main():
    server = AsyncServer("0.0.0.0", 10400)
    await server.run(OV_WakeWord_Handler)

if __name__ == "__main__":
    asyncio.run(main())
