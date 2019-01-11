import os
from fronted.mandarin import mtts
'''
你需要准备的:
    1. 语音文件路径。包含.wav音频文件
    2. 文本文件。 单独的文件，里面的每一行示例为 "0001   今天天气不错啊", 前面的0001 要在音频文件路径下有对应的 0001.wav文件,后面的内容是音频对应的中文内容
    3. 声学模型文件。此模型是一个语音识别模型，用kaldi训练得到的
须知： 中文TTS，其实使用了一个语音识别模型来自动标注音频文件，此项目标注到了音素级别，此项目的音素选取的是声韵母。如果你的数据已经标注到音素级别，不需要此目录

'''
if __name__ == "__main__":

    wav_path = ""
    textfile = ""
    acoustic_model_path = ""
    output_path = ""
    mtts.generate_label(textfile, wav_path, output_path,
                   acoustic_model_path)