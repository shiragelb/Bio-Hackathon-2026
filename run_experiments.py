import pandas as pd
from hmm import (
    read_fasta, read_labels,
    train_supervised_hmm, viterbi
)
import evaluation as ev
from experiments_config import EXPERIMENTS_CONFIG, MODEL_CONFIG


def run_pipeline(model):
    all_results = [] 
    config = MODEL_CONFIG[model]
    base_path = config["base_path"]

    for exp in EXPERIMENTS_CONFIG:
        for train_group_name, train_strains_list in exp['train'].items():
            for test_group_name, test_strains_list in exp['test'].items():
                exp_name = f"Train: {train_group_name}, Test: {test_group_name}"
                train_files = [
                    (f"{strain}_genome.fasta", f"{strain}_labels.fasta") 
                    for strain in train_strains_list
                ]
                
                test_files = [
                    (f"{strain}_genome.fasta", f"{strain}_labels.fasta") 
                    for strain in test_strains_list
                ]

                print(f"\n--- Running Experiment: {exp_name} ---")
                
                train_seq = ""
                train_labels = ""
                try:
                    for genome_file, label_file in train_files:
                        train_seq += read_fasta(f"{base_path}/{genome_file}")
                        train_labels += read_labels(f"{base_path}/{label_file}")
                    
                    print(f"Training on {len(train_seq)} bp...")
                    emissions, transitions, init = train_supervised_hmm(train_seq, train_labels,config["states"], config["alphabet"], config["laplace_smoothing"])
                    
                    for test_genome, test_label_file in test_files:
                        print(f"Testing on {test_genome}...")
                        
                        t_seq = read_fasta(f"{base_path}/{test_genome}")
                        t_true = read_labels(f"{base_path}/{test_label_file}")
                                                
                        t_pred = viterbi(t_seq, emissions, transitions, init, config["states"])

                        clean_genome_name = test_genome.replace("_genome.fasta", "")
                        metrics = ev.get_metrics_dict(t_true, t_pred, genome_name=clean_genome_name)
                        
                        metrics['Category'] = exp_name 
                        
                        all_results.append(metrics)
                        print(f"  > {clean_genome_name}: F1 = {metrics['F1_Score']:.4f}")

                except Exception as e:
                    print(f"Error in experiment {exp_name}: {e}")

    print("\nGenerating summary plots...")
    
    if all_results:
        df_res = pd.DataFrame(all_results)
        print(df_res[['Category', 'Genome', 'F1_Score', 'IoU']])
        
        ev.plot_grouped_benchmark(all_results, metric='F1_Score') 
        ev.plot_confusion_matrix_grid(all_results)                
        ev.plot_violin_distributions(all_results, metric='F1_Score')
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_pipeline("basic")