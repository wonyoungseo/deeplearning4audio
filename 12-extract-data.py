import os
import math
import json
import librosa

DATASET_PATH = "datasets/genres"
JSON_PATH = "datasets/processed/data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(
    dataset_path, json_path, num_mfcc=13, num_fft=2048, hop_length=512, num_segments=5
):

    """
    Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :param: num_segments (int): Number of segments we want to divide sample tracks into
    :return:
    """

    data = {"mapping": [], "labels": [], "mfcc": []}

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segments = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            for f in filenames:

                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    end = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        signal[start:end],
                        sample_rate,
                        n_mfcc=num_mfcc,
                        n_fft=num_fft,
                        hop_length=hop_length,
                    )
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segments:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print(f"{file_path}, segment: {d+1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
