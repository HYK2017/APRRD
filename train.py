import json
from preparation import setup_device, initialize_models, load_noisy_dataset, setup_training, train

with open("./config.json", "r") as f:
    config = json.load(f)
                
if __name__ == "__main__":
    device, device_ids = setup_device()
    BSN_info, NBSN_info = initialize_models(device, device_ids, training=True)
    BSN_tr_info, NBSN_tr_info = setup_training(BSN_info[0], NBSN_info[0]) # ['SIDD_Medium']
    
    dataset_list = list(config["dataset_path"].keys())[0:1]
    for set_name in dataset_list:
        dataset_info = load_noisy_dataset(set_name)
        train(dataset_info, BSN_info, BSN_tr_info, device)
        train(dataset_info, NBSN_info, NBSN_tr_info, device)