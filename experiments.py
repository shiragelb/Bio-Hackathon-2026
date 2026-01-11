"""
EXPERIMENT1:
Train on closely related, non-pathogenic E. coli laboratory and commensal strains,
and evaluate generalization to pathogenic E. coli strains and to related enteric species.
"""
EXPERIMENT1 = {
    "train": {"ecoli_basic":
              ["K12_MG1655", "K12_W3110", "BW25113","B_REL606", "HS"]},  # similar strains (lab + commensal)
    "test": 
        {"similar_ecoli_test":
        ["Escherichia_coli_CFT073", "Escherichia_coli_UTI89"],          
        # tests generalization to moderately pathogenic E. coli (UPEC)
        "different_pathogens_ecoli_test":
        ["Escherichia_coli_O157_H7_Sakai", "Escherichia_coli_O104_H4"], 
        # tests generalization to highly pathogenic E. coli strains
        "salmonella_test":
        ["Salmonella_enterica Typhimurium 14028S", "Salmonella_enterica Typhimurium LT2"],    
        # tests cross-species generalization to a related but more distant enteric species
        "shigella_test":
        ["Shigella_Shigella flexneri 2a 301", "Shigella_Shigella sonnei Ss046"]  
        # tests cross-species generalization to very closely related E. coli-like bacteria
}
}


EXPERIMENT2 = {
    "train": {"ecoli_diverse_pathogenic_train": ["CFT073", "UTI89", "O157_H7_Sakai", "O127_H6", "O104_H4", "SE11", "042"]},
    "test": {
        "ecoli_k12_lab_test": ["K12_MG1655", "K12_W3110"],  # lab K-12
        "ecoli_lab_derivatives_test": ["BW25113", "B_REL606"],      # lab derivatives
       "salmonella_test": ["Salmonella_enterica Typhimurium 14028S", "Salmonella_enterica Typhimurium LT2"],    # Salmonella
        "shigella_test": ["Shigella_Shigella flexneri 2a 301", "Shigella_Shigella sonnei Ss046"]  # Shigella
}
}

EXPERIMENT3 = {
    "train": {"pathogenic_group": ["CFT073", "UTI89", "536"]},
    "test": {
        "same_pathogenic": ["UMN026"],  # same
        "different_pathogenic": ["Sakai_O157H7", "O127-H6", "K12-MG1655"]  # different
    }
}

EXPERIMENT4 = {
    "train": {"shigella + ecoli": ["Shigella flexneri 2a 301", "Shigella sonnei Ss046", "K12-MG1655", "HS"]},
    "test": {
        "salmonella_test": ["Salmonella_enterica Typhimurium 14028S", "Salmonella_enterica Typhimurium LT2"],
    }
}
