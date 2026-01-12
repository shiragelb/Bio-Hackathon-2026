from pathlib import Path
import pandas as pd
from hmm import (
    read_fasta, read_labels
)
import importlib
import evaluation as ev
from experiments_config import EXPERIMENTS_CONFIG, MODEL_CONFIG, EXPERIMENTS_CONFIG_TEMP


def run_pipeline(model):
    all_results = [] 
    config = MODEL_CONFIG[model]
    base_path = config["base_path"]

    try:
        model_module = importlib.import_module(config["module_name"])
        train_func = getattr(model_module, config["train_func"])
        viterbi_func = getattr(model_module, config["viterbi_func"])
        
        print(f"Loaded '{config['train_func']}' and '{config['viterbi_func']}' from {config['module_name']}")
        
    except (ImportError, AttributeError) as e:
        print(f"Critical Error: Could not load model functions for {model}. {e}")
        return
    
    for exp in EXPERIMENTS_CONFIG:
        results = [] 
        gene_length_data = []
        non_coding_data = []
        for train_group_name, train_strains_list in exp['train'].items():
            print(f"\n=== Training Group: {train_group_name} ===")
            train_files = [
                    (f"{strain}_genome.fasta", f"{strain}_labels.fasta") 
                    for strain in train_strains_list
            ]
            train_seq = ""
            train_labels = ""
            try:
                for genome_file, label_file in train_files:
                    train_seq += read_fasta(f"{base_path}/{genome_file}", alphabet=config["alphabet"])
                    train_labels += read_labels(f"{base_path}/{label_file}", states=config["states"])
                
                print(f"Training on {len(train_seq)} bp...")
                emissions, transitions, init = train_func(train_seq, train_labels, config["states"], config["alphabet"])
            except Exception as e:
                print(f"Error during training for group {train_group_name}: {e}")
                continue
                

            for test_group_name, test_strains_list in exp['test'].items():
                exp_name = f"Train: {train_group_name}, Test: {test_group_name}"
                
                test_files = [
                    (f"{strain}_genome.fasta", f"{strain}_labels.fasta") 
                    for strain in test_strains_list
                ]
           
                try:
                    for test_genome, test_label_file in test_files:
                        print(f"Testing on {test_genome}...")
                        
                        t_seq = read_fasta(f"{base_path}/{test_genome}", alphabet=config["alphabet"])
                        t_true = read_labels(f"{base_path}/{test_label_file}", states=config["states"])
                                                
                        t_pred = viterbi_func(t_seq, emissions, transitions, init, config["states"], config["alphabet"])

                        length_stats = ev.collect_gene_length_stats(t_true, t_pred)
                        
                        for item in length_stats:
                            item['Experiment'] = exp_name
                            gene_length_data.append(item)

                        nc_stats = ev.collect_non_coding_length_stats(t_true, t_pred)
                        for item in nc_stats:
                            item['Experiment'] = exp_name
                            non_coding_data.append(item)
                    
                        clean_genome_name = test_genome.replace("_genome.fasta", "")
                        metrics = ev.get_metrics_dict(t_true, t_pred, genome_name=clean_genome_name)
                        metrics['Train'] = train_group_name
                        metrics['Test'] = test_group_name
                        metrics['Experiment'] = exp_name 
                        metrics['Model'] = model
                        
                        results.append(metrics)
                        all_results.append(metrics)
                        print(f"  > {clean_genome_name}: F1 = {metrics['F1_Score']:.4f}")

                except Exception as e:
                    print(f"Error in experiment {exp_name}: {e}")

            print("\nGenerating summary plots...")
            
            if results:
                df_res = pd.DataFrame(results)
                print(df_res[['Train', 'Test', 'Precision', 'Recall', 'F1_Score']])
                
                path = f"plots/{model}/{train_group_name}"
                Path(path).mkdir(parents=True, exist_ok=True)
                ev.plot_grouped_benchmark(results, metric='F1_Score', save_path=path) 
                ev.plot_confusion_matrix_grid(results, save_path=path)                
                if gene_length_data:
                    print("Plotting Success by Gene Length...")
                    ev.plot_success_by_length_binned(gene_length_data, save_path=path)
                if non_coding_data:
                    print("Plotting Success by Non-Coding Length...")
                    ev.plot_non_coding_success_by_length(non_coding_data, save_path=path)
            else:
                print("No results generated.")

if __name__ == "__main__":
    run_pipeline("more_hidden_states")