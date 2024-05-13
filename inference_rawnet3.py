import argparse
import itertools
import os
import sys
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import tqdm
import pandas as pd

from utils.RawNetModel import RawNet3
from utils.RawNetBasicBlock import Bottle2neck

def main(args: Dict) -> None:
    # 定义RawNet3模型与参数
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    gpu = False

    # 预训练模型加载
    model.load_state_dict(
        torch.load(
            "./pretrain/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    model.eval()
    print("RawNet3 initialised & weights loaded!")

    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    df_test = pd.read_csv(args.test_path)
    test_list = df_test["wav_path"].tolist()
    outputs = []
    # 提取说话人embedding
    for idx, file in tqdm.tqdm(enumerate(test_list), total = len(test_list)):
        output = extract_speaker_embd(
            model,
            fn=file,
            n_samples=48000, # 16k*3s
            n_segments=args.n_segments,
            gpu=gpu,
        ).mean(0)
        # print(f"file: {file}, output.shape: {output.shape}")
        outputs.append(output)
    outputs = np.array(outputs.cpu().numpy())
    np.save(args.out_dir, outputs)
    df_result = pd.DataFrame(outputs, columns=["speech_name", "pred_label"])
    df_result.to_csv(args.save_path, index=False, header=None)
    return

# 提取说话人embedding
# 五个输入: model - RawNet3模型, fn - 输入音频文件路径, n_samples - 采样点数, n_segments - 分段数, gpu - 是否使用GPU
def extract_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    # 检查是否单声道
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
        )
    # 检查音频采样率是否是16k
    if sample_rate != 16000:
        raise ValueError(
            f"RawNet3 supports 16k sampling rate only. Input data's sampling rate is {sample_rate}."
        )
    # 音频长度不足时进行填充
    if (
        len(audio) < n_samples
    ):  # RawNet3 was trained using utterances of 3 seconds
        shortage = n_samples - len(audio) + 1
        audio = np.pad(audio, (0, shortage), "wrap")

    audios = []
    startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    for asf in startframe:
        audios.append(audio[int(asf) : int(asf) + n_samples])

    audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
    if gpu:
        audios = audios.to("cuda")
    with torch.no_grad():
        output = model(audios)
        # print(output.shape)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RawNet3 inference")

    parser.add_argument(
        "--test_path",
        type=str,
        default="./data/finvcup9th_1st_ds4_16k/finvcup9th_1st_ds4_16k_test_data.csv",
        help="Input file to extract embedding. Required when 'inference_utterance' is True",
    )
    parser.add_argument("--out_dir", type=str, default="./out.npy")
    parser.add_argument(
        "--n_segments",
        type=int,
        default=10,
        help="number of segments to make using each utterance",
    )
    parser.add_argument('--save_path', type=str, default='./submit/submit_rawnet.csv', help='Path of result')
    
    args = parser.parse_args()
    main(args)
    # sys.exit(main(args))