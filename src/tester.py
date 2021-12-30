import logging
from nussl.evaluation import BSSEvalScale
from nussl import AudioSignal
logging.getLogger(__name__)


def hi(name):
    logging.info(f"hi {name}")


def load_as_AudioSignal_8000(ll: list):
    return [AudioSignal(_, sample_rate=8000) for _ in ll]


def do_estimate():

    true_s1 = "./dataset/senpai_data/heart_lung_sam2/mix/training_clean_心跳/0dB/4_1.wav"
    true_s2 = "./dataset/senpai_data/heart_lung_sam2/mix/training_noise_呼吸/0dB/4_1.wav"

    esti_s1 = "./log/DAE_C_2021_1227_1609_34/test_source1/4_1.wav"
    esti_s2 = "./log/DAE_C_2021_1227_1609_34/test_source2/4_1.wav"

    true_sources_list = load_as_AudioSignal_8000([true_s1, true_s2])
    estimated_sources_list = load_as_AudioSignal_8000([esti_s1, esti_s2])

    esti = BSSEvalScale(true_sources_list, estimated_sources_list,
                        compute_permutation=True,
                        best_permutation_key='SDR')  # Default:SDR

    for key, item in esti.evaluate().items():
        if isinstance(item, dict):
            # print(key, ":")
            # for kkey, iitem in item.items():
            #     print('\t', kkey, ': ', iitem[0])
            print("========================")
            print("SDR:", item['SI-SDR'][0])
            print("SIR:", item['SI-SIR'][0])
            print("SAR:", item['SI-SAR'][0])
            print("SNR:", item['SNR'][0])


        else:
            print(key, ":", item)


if __name__ == "__main__":
    do_estimate()




