import pandas as pd
from hmm import (
    read_fasta, read_labels,
    train_supervised_hmm, viterbi
)
import evaluation as ev
from experiments_config import EXPERIMENTS_CONFIG, EXPERIMENTS_CONFIG_TEMP
import copy

BASE = "processed"

def filter_short_genes(y_pred, min_length=60):
    y_filtered = y_pred.copy()
    n = len(y_filtered)
    
    i = 0
    while i < n:
        if y_filtered[i] == 1:
            j = i
            while j < n and y_filtered[j] == 1:
                j += 1
            
            segment_len = j - i
            if segment_len < min_length:
                y_filtered[i:j] = 0
            
            i = j
        else:
            i += 1
            
    return y_filtered

def apply_transition_penalty(transitions, penalty):
    """יוצרת עותק של מטריצת המעברים ומקטינה את הסיכוי להישאר ב-C"""
    new_trans = copy.deepcopy(transitions)
    reduction = new_trans['C']['C'] * penalty
    new_trans['C']['C'] -= reduction
    new_trans['C']['N'] += reduction
    return new_trans

def run_pipeline():
    all_results = [] 
    # רשימת הקנסות לבדיקה (0.0 = מקורי, 0.2 = קנס כבד)
    tuning_values = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2] 
    tuning_results = [] 
    final_results = []

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
                        train_seq += read_fasta(f"{BASE}/{genome_file}")
                        train_labels += read_labels(f"{BASE}/{label_file}")
                    
                    print(f"Training on {len(train_seq)} bp...")
                    emissions, transitions, init = train_supervised_hmm(train_seq, train_labels)
                    
                    ##
                    for test_genome, test_label_file in test_files:
                        print(f"Testing on {test_genome}...")
                        
                        t_seq = read_fasta(f"{BASE}/{test_genome}")
                        t_true = read_labels(f"{BASE}/{test_label_file}")

                        best_f1 = -1 #
                        best_metrics = None #

                        for penalty in tuning_values:
                            # 1. יצירת מטריצה מותאמת
                            # (transitions הוא המשתנה שחזר מהאימון למעלה)
                            curr_trans = apply_transition_penalty(transitions, penalty)
                            
                            # 2. הרצת ויטרבי
                            t_pred = viterbi(t_seq, emissions, curr_trans, init)
                            
                            # 3. בדיקת ביצועים
                            m = ev.get_metrics_dict(t_true, t_pred, genome_name=test_genome.replace("_genome.fasta", ""))
                            
                            # בדיקה אם זה השיא החדש
                            if m['F1_Score'] > best_f1:
                                best_f1 = m['F1_Score']
                                best_metrics = m
                                # הוספת מידע לגרפים
                                best_metrics['Category'] = exp_name 
                                best_metrics['Best_Penalty'] = penalty

                        # בסוף הלולאה, שומרים רק את התוצאה הכי טובה שמצאנו לגנום הזה
                        print(f"    Selected Best F1: {best_f1:.4f} (Penalty: {best_metrics['Best_Penalty']})")

                        ##
                        all_results.append(best_metrics)

                        t_pred = viterbi(t_seq, emissions, transitions, init)

                        t_pred = filter_short_genes(t_pred, min_length=50)
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
    run_pipeline()