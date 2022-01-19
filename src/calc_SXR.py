import logging
from nussl.evaluation import BSSEvalScale
from nussl import AudioSignal
logging.getLogger(__name__)
import json

def hi(name):
    logging.info(f"hi {name}")


def load_as_AudioSignal_8000(ll: list):
    return [AudioSignal(_, sample_rate=8000) for _ in ll]


def do_estimate(trur_s: list, esti_s: list):

    true_s1 = trur_s[0]
    true_s2 = trur_s[1]

    esti_s1 = esti_s[0]
    esti_s2 = esti_s[1]

    true_sources_list = load_as_AudioSignal_8000([true_s1, true_s2])
    estimated_sources_list = load_as_AudioSignal_8000([esti_s1, esti_s2])

    esti = BSSEvalScale(true_sources_list, estimated_sources_list,
                        compute_permutation=True,
                        best_permutation_key='SDR')  # Default:SDR

    for key, item in esti.evaluate().items():
        if isinstance(item, dict):
            print("========================")
            print("SDR:", item['SI-SDR'][0])
            print("SIR:", item['SI-SIR'][0])
            print("SAR:", item['SI-SAR'][0])
            print("SNR:", item['SNR'][0])
    else:
            _ = json.dumps(item,
                      sort_keys=True,
                      indent=2
                      )
            print("========================")
            print(_)


if __name__ == "__main__":
    """
    計算 混合源分離後與 原始 source 的分數
    分數越大越好
    """
    # 原始無混 音源 A, B
    true_s1 = "./dataset/senpai_data/heart_lung_sam2/mix/training_clean_心跳/0dB/4_1.wav"
    true_s2 = "./dataset/senpai_data/heart_lung_sam2/mix/training_noise_呼吸/0dB/4_1.wav"
    trur_s = [true_s1, true_s2]  # [心, 肺]

    # 做盲源分離後的 分離源 對應於 A, B無混音源，有序 不得弄混
    esti_s1 = "./log/DAE_C_2022_0120_0145_37/test_my_source1/4_1.wav"
    esti_s2 = "./log/DAE_C_2022_0120_0145_37/test_my_source0/4_1.wav"
    #esti_s = [esti_s2, esti_s1]  # [心, 肺]
    esti_s = [esti_s1, esti_s2]  # [心, 肺]

    do_estimate(trur_s, esti_s)




