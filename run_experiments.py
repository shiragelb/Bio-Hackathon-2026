import pandas as pd
from hmm import (
    read_fasta, read_labels,
    train_supervised_hmm, viterbi
)
import evaluation as ev
from experiments_config import EXPERIMENTS_CONFIG

BASE = "processed_data"

def run_pipeline():
    all_results = [] 

    for exp in EXPERIMENTS_CONFIG:
        for train_exp in exp['train']:
            for test_exp in exp['test']:
                exp_instance = {
                    'name': f"Train: {train_exp['name']}, Test: {test_exp['name']}",
                    'train': [
                        (f"{train_exp['name']}_genome.fasta", f"{train_exp['name']}_labels.fasta")
                    ],
                    'test': [
                        (f"{test_exp['name2']}_genome.fasta", f"{test_exp['name2']}_labels.fasta")
                    ]
                }


                print(f"\n--- Running Experiment: {exp_instance['name']} ---")
                
                train_seq = ""
                train_labels = ""
                try:
                    for genome_file, label_file in exp_instance['train']:
                        train_seq += read_fasta(f"{BASE}/{genome_file}")
                        train_labels += read_labels(f"{BASE}/{label_file}")
                    
                    print(f"Training on {len(train_seq)} bp...")
                    emissions, transitions, init = train_supervised_hmm(train_seq, train_labels)
                    
                    for test_genome, test_label_file in exp_instance['test']:
                        print(f"Testing on {test_genome}...")
                        
                        t_seq = read_fasta(f"{BASE}/{test_genome}")
                        t_true = read_labels(f"{BASE}/{test_label_file}")
                                                
                        t_pred = viterbi(t_seq, emissions, transitions, init)
                        

                        metrics = ev.get_metrics_dict(t_true, t_pred, genome_name=test_genome)
                        metrics['Category'] = exp_instance['name'] 
                        all_results.append(metrics)
                        
                        print(f"Result for {test_genome}: F1 = {metrics['F1_Score']:.4f}")
        
                        ev.plot_genome_segment(t_true, t_pred, start=20000, end=22000, 
                                            title=f"{exp_instance['name']} - {test_genome}")

                except Exception as e:
                    print(f"Error in experiment {exp_instance['name']}: {e}")


    print("\nGenerating summary plots...")
    
    df_res = pd.DataFrame(all_results)
    print(df_res[['Category', 'Genome', 'F1_Score', 'IoU']])
    
    ev.plot_all_experiments(all_results, metric='F1_Score')
    ev.plot_all_experiments(all_results, metric='IoU')

if __name__ == "__main__":
    run_pipeline()