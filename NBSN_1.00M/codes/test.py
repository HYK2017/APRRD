import json
from preparation import setup_device, initialize_models, load_noisy_dataset, inference

with open("./config.json", "r") as f:
    config = json.load(f)  
    
if __name__ == "__main__":
    device, device_ids = setup_device()
    
    trainingsets_name = list(config["dataset_path"]["training"].keys())   # ["SIDD_Medium", "SIDD_Benchmark", "DND_Benchmark"]
    testsets_name = list(config["dataset_path"]["inference"].keys())      # ["SIDD_Validation", "SIDD_Benchmark", "DND_Benchmark"]
    
    for trainingset_name in trainingsets_name[0:1]:
        # This setting checks the testset results for the model trained on SIDD_medium.
        # If you want to check the fully self-supervised result, adjust the indexing of trainingsets_name accordingly.
        
        # If you want to test using the model you have trained, change the name of the folder where the trained model is stored from trained to pretrained.
        BSN_info, NBSN_info = initialize_models(device, device_ids, trainingset_name)
        for testset_name in testsets_name:
            if trainingset_name != "SIDD_Medium" and trainingset_name != testset_name:
                continue
            testset_info = load_noisy_dataset(testset_name, training=False)
            inference(trainingset_name, testset_info, BSN_info, device)      # Get the result of BSN
            inference(trainingset_name, testset_info, NBSN_info, device)     # Get the result of NBSN