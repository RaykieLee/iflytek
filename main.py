# coding:utf-8
import os
import markdown2
from whisper_online import *

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
try:
    from dotenv import load_dotenv
except ImportError:
    raise RuntimeError('Python environment for SPARK AI is not completely set up: required package "python-dotenv" is missing.') from None
src_lan = "en"  # source language
tgt_lan = "en"  # target language  -- same as source for ASR, "en" if translate task is used

load_dotenv()
def test_stream():
    from sparkai.core.callbacks import StdOutCallbackHandler
    spark = ChatSparkLLM(
        spark_api_url=os.environ["SPARKAI_URL"],
        spark_app_id=os.environ["SPARKAI_APP_ID"],
        spark_api_key=os.environ["SPARKAI_API_KEY"],
        spark_api_secret=os.environ["SPARKAI_API_SECRET"],
        spark_llm_domain=os.environ["SPARKAI_DOMAIN"],
        streaming=False,
    )
    messages = []
    while True:
        user_input = input("请输入您的问题: ")
        messages.append(ChatMessage(role="user", content=user_input))
        handler = ChunkPrintHandler()
        a = markdown2.markdown(spark.generate([messages], callbacks=[handler]).generations[0][0].text)
        print(a)
        messages.append(ChatMessage(role="assistant", content=a))


if __name__ == '__main__':
    asr = FasterWhisperASR('zh', "tiny")  # loads and wraps Whisper model
    # set options:
    # asr.set_translate_task()  # it will translate from lan into English
    # asr.use_vad()  # set using VAD

    online = OnlineASRProcessor(asr)  # create processing object with default buffer trimming option

    # while audio_has_not_ended:  # processing loop:
    #     a =  ''# receive new audio chunk (and e.g. wait for min_chunk_size seconds first, ...)
    #     online.insert_audio_chunk(a)
    #     o = online.process_iter()
    #     print(o)  # do something with current partial output
    # at the end of this audio processing
    o = online.finish()
    print(o)  # do something with the last output

    online.init()  # refresh if you're going to re-use the object for the next audio
    test_stream()