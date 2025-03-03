import json
from preparation import setup_device, initialize_models, load_noisy_dataset, setup_training, train

with open("./config.json", "r") as f:
    config = json.load(f)
                
if __name__ == "__main__":
    device, device_ids = setup_device()
    
    trainingsets_name = list(config["dataset_path"]["training"].keys())   # ["SIDD_Medium", "SIDD_Benchmark", "DND_Benchmark"]
    
    BSN_info, NBSN_info = initialize_models(device, device_ids)
    BSN_tr_info, NBSN_tr_info = setup_training(BSN_info[0], NBSN_info[0])
    
    for trainingset_name in trainingsets_name[0:1]:
        # This setting trains only on SIDD_medium.
        # If you want to train on SIDD_Benchmark or DND_Benchmark, adjust the indexing of trainingsets_name accordingly.
        trainingsets_info = load_noisy_dataset(trainingset_name, training=True)
        
        train(trainingsets_info, BSN_info, BSN_tr_info, device)              # BSN training
        train(trainingsets_info, NBSN_info, NBSN_tr_info, device)            # NBSN distillation