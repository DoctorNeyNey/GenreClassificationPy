#!/usr/bin/python3.8
import os
import librosa
import math
import json

DATASET_PATH = "genres_big"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050 #customary value 
DURATION = 30 #measured in seconds (given in data set
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    #dictionary to store data
    data = {
        "mapping": [], #genre tags
        "mfcc": [], #training input
        "labels": [] #targets/expect output
    }
   
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    print(num_samples_per_segment)
    exit()
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
    print(SAMPLES_PER_TRACK, num_samples_per_segment, expected_num_mfcc_vectors_per_segment)
    #loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #ensure that we're not at the root level
        if dirpath is not dataset_path:
            
            #save the semantic label
            dirpath_components = dirpath.split("/") #genre/blues
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            #process fils for a specific genre
            for f in filenames:
                #load audio file
                try:
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    print(f , "in genre", dirpath, "wih", num_segments, "segments")
                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s 
                        finish_sample = start_sample + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        #store mfcc for segment if it has the expected length
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print(len(data["mfcc"]))
                            #print("{}, segment:{}".format(file_path, s))

                except:
                    print(f , "file {} had an error. Skipping this file".format(f))
    print("Dumping Into Json File: {}".format(json_path))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("File Created")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)








