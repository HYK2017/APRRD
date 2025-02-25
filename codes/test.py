import json
from preparation import setup_device, initialize_models, load_noisy_dataset, inference

with open("./config.json", "r") as f:
    config = json.load(f)  
    
if __name__ == "__main__":
    device, device_ids = setup_device()
    BSN_info, NBSN_info = initialize_models(device, device_ids, training=False)
    
    dataset_list = list(config["dataset_path"].keys())[1:]  # ['SIDD_Validation', 'SIDD_Benchmark', 'DND']
    for set_name in dataset_list:
        dataset_info = load_noisy_dataset(set_name)
        inference(dataset_info, BSN_info, device)  # Get BSN results for ['SIDD_Validation', 'SIDD_Benchmark', 'DND']
        inference(dataset_info, NBSN_info, device)  # Get NBSN results for ['SIDD_Validation', 'SIDD_Benchmark', 'DND']