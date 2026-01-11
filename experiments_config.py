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
                ["E.coli_CFT073", "E.coli_UTI89"],          
                # tests generalization to moderately pathogenic E. coli (UPEC)
                "ecoli_intestinal_pathogens":
                ["E.coli_O157_H7 Sakai", "E.coli_O104_H4"], 
                # tests generalization to highly pathogenic E. coli strains
                "salmonella":
                ["Salmonella enterica_Typhimurium 14028S", "Salmonella enterica_Typhimurium LT2"],    
                # tests cross-species generalization to a related but more distant enteric species
                "shigella":
                ["Shigella_Shigella flexneri 2a 301", "Shigella_Shigella sonnei Ss046"]  
                # tests cross-species generalization to very closely related E. coli-like bacteria
        }
    },
    {
        "train": {
            "ecoli_diverse_pathogen": 
            ["E.coli_CFT073", "E.coli_UTI89", "E.coli_O157_H7 Sakai", "E.coli_O127-H6", "E.coli_O104_H4", "E.coli_SE11", "E.coli_042"]
        },
        "test": {
            "ecoli_k12_lab": 
                ["E.coli_K12_MG1655", "E.coli_K12_W3110"],  # lab K-12
            "ecoli_lab_derivatives": 
                ["E.coli_BW25113", "E.coli_B REL606"],      # lab derivatives
            "salmonella": 
                ["Salmonella enterica_Typhimurium 14028S", "Salmonella enterica_Typhimurium LT2"],    # Salmonella
            "shigella": 
                ["Shigella_Shigella flexneri 2a 301", "Shigella_Shigella sonnei Ss046"]  # Shigella
        }
    },
    {
        "train": {
            "ecoli_upec": 
                ["E.coli_CFT073", "E.coli_UTI89", "E.coli_536"]
        },
        "test": {
            "ecoli_upec": 
                ["E.coli_UMN026"],  # same
            "ecoli_intestinal_pathogens": 
                ["E.coli_O157_H7 Sakai", "E.coli_O127-H6", "E.coli_K12_MG1655"]  # different
        }
    },
    {
        "train": {
            "shigella_ecoli": 
                ["Shigella_flexneri 2a 301", "Shigella_sonnei Ss046", "E.coli_K12_MG1655", "E.coli_HS"]
        },
        "test": {
            "salmonella": 
                ["Salmonella enterica_Typhimurium 14028S", "Salmonella enterica_Typhimurium LT2"],
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