"""
EXPERIMENT1:
Train on closely related, non-pathogenic E. coli laboratory and commensal strains,
and evaluate generalization to pathogenic E. coli strains and to related enteric species.
"""
EXPERIMENTS_CONFIG = [
    {
        "train": {
            "ecoli_commensal_lab":
              ["K12_MG1655", "K12_W3110", "BW25113","B_REL606", "HS"]
        },  # similar strains (lab + commensal)
        "test": {
            "ecoli_upec_pathogens":
                ["Escherichia_coli_CFT073", "Escherichia_coli_UTI89"],          
                # tests generalization to moderately pathogenic E. coli (UPEC)
                "ecoli_intestinal_pathogens":
                ["Escherichia_coli_O157_H7_Sakai", "Escherichia_coli_O104_H4"], 
                # tests generalization to highly pathogenic E. coli strains
                "salmonella":
                ["Salmonella_enterica Typhimurium 14028S", "Salmonella_enterica Typhimurium LT2"],    
                # tests cross-species generalization to a related but more distant enteric species
                "shigella":
                ["Shigella_Shigella flexneri 2a 301", "Shigella_Shigella sonnei Ss046"]  
                # tests cross-species generalization to very closely related E. coli-like bacteria
        }
    },
    {
        "train": {
            "ecoli_diverse_pathogen": 
            ["CFT073", "UTI89", "O157_H7_Sakai", "O127_H6", "O104_H4", "SE11", "042"]
        },
        "test": {
            "ecoli_k12_lab": 
                ["K12_MG1655", "K12_W3110"],  # lab K-12
            "ecoli_lab_derivatives": 
                ["BW25113", "B_REL606"],      # lab derivatives
            "salmonella": 
                ["Salmonella_enterica Typhimurium 14028S", "Salmonella_enterica Typhimurium LT2"],    # Salmonella
            "shigella": 
                ["Shigella_Shigella flexneri 2a 301", "Shigella_Shigella sonnei Ss046"]  # Shigella
        }
    },
    {
        "train": {
            "ecoli_upec": 
                ["CFT073", "UTI89", "536"]
        },
        "test": {
            "ecoli_upec": 
                ["UMN026"],  # same
            "ecoli_intestinal_pathogens": 
                ["Sakai_O157H7", "O127-H6", "K12-MG1655"]  # different
        }
    },
    {
        "train": {
            "shigella_ecoli": 
                ["Shigella flexneri 2a 301", "Shigella sonnei Ss046", "K12-MG1655", "HS"]
        },
        "test": {
            "salmonella": 
                ["Salmonella_enterica Typhimurium 14028S", "Salmonella_enterica Typhimurium LT2"],
        }
    }
]


EXPERIMENTS_CONFIG_TEMP = [
    {
        "train": {
            "ecoli_temp":
              ["Escherichia_coli_K12-MG1655", "Escherichia_coli_E. coli B REL606", "Escherichia_coli_HS","Escherichia_coli_SE11"]
        },
        "test": {
            "ecoli_temp_test":
                ["Escherichia_coli_042"],          
        }
    }
]