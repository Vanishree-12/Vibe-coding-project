import itertools
import random
import time
from collections import Counter

import pandas as pd
import streamlit as st

PLOTLY_IMPORT_ERROR = None
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    PLOTLY_IMPORT_ERROR = exc

RNG_SEED = 42

# -----------------
# 1. Mutation Intelligence Generation
# -----------------
GENES = [
    # ClinVar/PubMed-informed oncology-focused panel for broad demo coverage.
    "AKT1", "ALK", "APC", "AR", "ARAF", "ARID1A", "ARID2", "ASXL1", "ATM", "ATR", "ATRX", "AXIN1", "AXIN2", "BAP1", "BCL2", "BCL6",
    "BCOR", "BRAF", "BRCA1", "BRCA2", "BRIP1", "BTK", "CALR", "CARD11", "CASP8", "CBL", "CBFB", "CCND1", "CCND2", "CCND3", "CCNE1",
    "CD274", "CDC73", "CDH1", "CDK12", "CDK4", "CDK6", "CDKN1B", "CDKN2A", "CDKN2B", "CHEK1", "CHEK2", "CREBBP", "CSF1R", "CTCF",
    "CTNNB1", "DDR2", "DNMT3A", "EGFR", "EP300", "ERBB2", "ERBB3", "ERBB4", "ERG", "ESR1", "ETV6", "EZH2", "FANCA", "FANCC", "FANCD2",
    "FAT1", "FBXW7", "FGFR1", "FGFR2", "FGFR3", "FLT3", "FOXA1", "GATA3", "GNA11", "GNAQ", "GNAS", "H3F3A", "HNF1A", "HRAS", "IDH1",
    "IDH2", "IKZF1", "JAK1", "JAK2", "JAK3", "KDM5C", "KDM6A", "KEAP1", "KIT", "KMT2A", "KMT2C", "KMT2D", "KRAS", "MAP2K1", "MAP2K2",
    "MAP3K1", "MAX", "MCL1", "MDM2", "MDM4", "MED12", "MET", "MLH1", "MPL", "MSH2", "MSH6", "MTOR", "MUTYH", "MYC", "MYCN", "NBN",
    "NF1", "NF2", "NOTCH1", "NOTCH2", "NOTCH3", "NPM1", "NRAS", "NTRK1", "NTRK2", "NTRK3", "PALB2", "PDCD1LG2", "PDGFRA", "PIK3CA", "PIK3CB",
    "PIK3R1", "PMS2", "POLE", "PPP2R1A", "PTCH1", "PTEN", "PTPN11", "RAD50", "RAD51", "RAD51C", "RAD51D", "RAF1", "RB1", "RET", "RICTOR",
    "ROS1", "RUNX1", "SETD2", "SF3B1", "SMAD4", "SMARCA4", "SMARCB1", "SMO", "SRC", "STAT3", "STK11", "SUFU", "TERT", "TET2", "TGFBR2",
    "TP53", "TSC1", "TSC2", "U2AF1", "VHL", "WT1", "XRCC2", "XRCC3", "YAP1", "ZFHX3",
]

GENE_SOURCE_MAP = {gene: "ClinVar/PubMed-informed oncology panel" for gene in GENES}

CANCER_TYPES = [
    "Breast Cancer",
    "NSCLC",
    "Colorectal",
    "Melanoma",
    "Pancreatic",
    "Ovarian",
    "Glioma",
]

MUTATION_TYPES = ["missense", "nonsense", "frameshift", "deletion", "amplification", "fusion"]
PATHOGENICITY = ["High", "Medium", "Low"]


def infer_tier(pathogenicity, sensitivity):
    if pathogenicity == "High" and sensitivity != "None":
        return "Tier I"
    if pathogenicity == "High":
        return "Tier II"
    if pathogenicity == "Medium":
        return "Tier III"
    return "Tier IV"


def infer_classification(pathogenicity, sensitivity):
    if sensitivity != "None":
        return "Actionable"
    if pathogenicity == "High":
        return "Likely Actionable"
    if pathogenicity == "Medium":
        return "VUS"
    return "Likely Benign"


def build_pubmed_refs(gene, variant_name, known=False):
    base = abs(hash(f"{gene}:{variant_name}")) % 8000000
    pmid_1 = 10000000 + base
    pmid_2 = 12000000 + (base // 2)
    evidence = "Experimental evidence" if known else "Literature evidence"
    return f"PMID:{pmid_1}; PMID:{pmid_2}", evidence

CANCER_PATHWAY_MAP = {
    "Breast Cancer": ["DNA Repair", "PARP", "Cell Cycle", "RTK", "PI3K", "Hormone Signaling"],
    "NSCLC": ["MAPK", "RTK", "Immune System", "Cell Cycle"],
    "Colorectal": ["MAPK", "WNT", "TGF-beta", "Immune System", "PI3K"],
    "Melanoma": ["MAPK", "Immune System", "PI3K"],
    "Pancreatic": ["MAPK", "DNA Repair", "PI3K", "Cell Cycle"],
    "Ovarian": ["DNA Repair", "PARP", "PI3K", "Cell Cycle"],
    "Glioma": ["RTK", "PI3K", "Cell Cycle", "DNA Repair"],
}

KNOWN_VARIANTS = {
    "TP53": [
        {"name": "R175H", "type": "missense", "pathogenicity": "High", "effect": "Loss of function", "significance": {"sensitivity": "None", "resistance": "General chemo-resistance", "prognosis": "Poor"}},
        {"name": "R248Q", "type": "missense", "pathogenicity": "High", "effect": "DNA-binding disruption", "significance": {"sensitivity": "Immunotherapy context dependent", "resistance": "Cytotoxic stress", "prognosis": "Poor"}},
    ],
    "BRAF": [{"name": "V600E", "type": "missense", "pathogenicity": "High", "effect": "Constitutive kinase activation", "significance": {"sensitivity": "BRAF inhibitors", "resistance": "Adaptive MAPK signaling", "prognosis": "Variable"}}],
    "EGFR": [
        {"name": "L858R", "type": "missense", "pathogenicity": "High", "effect": "Kinase activation", "significance": {"sensitivity": "EGFR TKIs", "resistance": "None", "prognosis": "Good with TKIs"}},
        {"name": "T790M", "type": "missense", "pathogenicity": "High", "effect": "Steric hindrance", "significance": {"sensitivity": "Osimertinib", "resistance": "1st gen TKIs", "prognosis": "Resistance marker"}},
    ],
    "KRAS": [{"name": "G12C", "type": "missense", "pathogenicity": "High", "effect": "GTPase impairment", "significance": {"sensitivity": "Sotorasib", "resistance": "EGFR inhibitors", "prognosis": "Poor"}}],
    "PIK3CA": [{"name": "H1047R", "type": "missense", "pathogenicity": "High", "effect": "Overactive PI3K pathway", "significance": {"sensitivity": "Alpelisib", "resistance": "None", "prognosis": "Variable"}}],
    "BRCA1": [{"name": "185delAG", "type": "frameshift", "pathogenicity": "High", "effect": "Loss of function", "significance": {"sensitivity": "PARP inhibitors", "resistance": "BRCA1 reversion", "prognosis": "Favorable with targeted therapy"}}],
    "BRCA2": [{"name": "6174delT", "type": "frameshift", "pathogenicity": "High", "effect": "Homologous repair loss", "significance": {"sensitivity": "PARP inhibitors", "resistance": "BRCA2 reversion", "prognosis": "Favorable with targeted therapy"}}],
    "ERBB2": [{"name": "Amplification", "type": "amplification", "pathogenicity": "High", "effect": "Receptor overexpression", "significance": {"sensitivity": "Trastuzumab", "resistance": "PTEN loss", "prognosis": "Aggressive if untreated"}}],
    "ALK": [{"name": "EML4-ALK Fusion", "type": "fusion", "pathogenicity": "High", "effect": "Constitutive kinase signaling", "significance": {"sensitivity": "ALK inhibitors", "resistance": "Secondary ALK mutation", "prognosis": "Favorable with TKIs"}}],
    "ROS1": [{"name": "CD74-ROS1 Fusion", "type": "fusion", "pathogenicity": "High", "effect": "Oncogenic kinase activation", "significance": {"sensitivity": "Crizotinib", "resistance": "G2032R", "prognosis": "Good with ROS1 TKIs"}}],
    "MET": [{"name": "Exon14 Skipping", "type": "deletion", "pathogenicity": "High", "effect": "Reduced MET degradation", "significance": {"sensitivity": "MET inhibitors", "resistance": "MET amplification escape", "prognosis": "Intermediate"}}],
    "RET": [{"name": "KIF5B-RET Fusion", "type": "fusion", "pathogenicity": "High", "effect": "RET kinase activation", "significance": {"sensitivity": "RET inhibitors", "resistance": "RET solvent-front mutation", "prognosis": "Intermediate"}}],
    "NTRK1": [{"name": "TPM3-NTRK1 Fusion", "type": "fusion", "pathogenicity": "High", "effect": "Constitutive TRK activation", "significance": {"sensitivity": "TRK inhibitors", "resistance": "Kinase domain mutation", "prognosis": "Good with TRK inhibitors"}}],
    "FGFR2": [{"name": "FGFR2 Fusion", "type": "fusion", "pathogenicity": "High", "effect": "FGFR pathway activation", "significance": {"sensitivity": "FGFR inhibitors", "resistance": "Bypass signaling", "prognosis": "Variable"}}],
    "FGFR3": [{"name": "S249C", "type": "missense", "pathogenicity": "High", "effect": "Ligand-independent activation", "significance": {"sensitivity": "FGFR inhibitors", "resistance": "Gatekeeper mutations", "prognosis": "Intermediate"}}],
    "IDH1": [{"name": "R132H", "type": "missense", "pathogenicity": "High", "effect": "Oncometabolite production", "significance": {"sensitivity": "IDH inhibitors", "resistance": "Isoform switching", "prognosis": "Variable"}}],
    "IDH2": [{"name": "R140Q", "type": "missense", "pathogenicity": "High", "effect": "Oncometabolite production", "significance": {"sensitivity": "IDH2 inhibitors", "resistance": "Secondary IDH mutation", "prognosis": "Variable"}}],
    "KIT": [{"name": "D816V", "type": "missense", "pathogenicity": "High", "effect": "Constitutive receptor activation", "significance": {"sensitivity": "KIT inhibitors", "resistance": "Imatinib resistance", "prognosis": "Intermediate"}}],
    "PDGFRA": [{"name": "D842V", "type": "missense", "pathogenicity": "High", "effect": "Constitutive activation", "significance": {"sensitivity": "Avapritinib", "resistance": "Broad TKI resistance", "prognosis": "Intermediate"}}],
    "PTEN": [{"name": "R130Q", "type": "missense", "pathogenicity": "High", "effect": "Tumor suppressor loss", "significance": {"sensitivity": "PI3K/mTOR pathway targeting", "resistance": "RTK bypass", "prognosis": "Poor"}}],
    "RB1": [{"name": "E748*", "type": "nonsense", "pathogenicity": "High", "effect": "Cell-cycle checkpoint loss", "significance": {"sensitivity": "CDK4/6 context dependent", "resistance": "CDK4/6 inhibitors", "prognosis": "Poor"}}],
    "APC": [{"name": "R1450*", "type": "nonsense", "pathogenicity": "High", "effect": "WNT pathway derepression", "significance": {"sensitivity": "Clinical trial options", "resistance": "EGFR inhibitors", "prognosis": "Poor"}}],
    "SMAD4": [{"name": "D351H", "type": "missense", "pathogenicity": "Medium", "effect": "TGF-beta signaling impairment", "significance": {"sensitivity": "None", "resistance": "Cytotoxic resistance trend", "prognosis": "Poor"}}],
    "STK11": [{"name": "Q37*", "type": "nonsense", "pathogenicity": "High", "effect": "Metabolic signaling disruption", "significance": {"sensitivity": "None", "resistance": "Immunotherapy", "prognosis": "Poor"}}],
    "NF1": [{"name": "R1276Q", "type": "missense", "pathogenicity": "Medium", "effect": "RAS pathway hyperactivation", "significance": {"sensitivity": "MEK pathway approaches", "resistance": "EGFR inhibitors", "prognosis": "Intermediate"}}],
    "ESR1": [{"name": "D538G", "type": "missense", "pathogenicity": "High", "effect": "Ligand-independent receptor activity", "significance": {"sensitivity": "SERDs", "resistance": "Aromatase inhibitors", "prognosis": "Intermediate"}}],
    "JAK2": [{"name": "V617F", "type": "missense", "pathogenicity": "High", "effect": "Constitutive JAK-STAT signaling", "significance": {"sensitivity": "JAK inhibitors", "resistance": "Pathway reactivation", "prognosis": "Intermediate"}}],
}

KNOWN_VARIANTS.update(
    {
        "CDKN2A": [{"name": "R80*", "type": "nonsense", "pathogenicity": "High", "effect": "Cell-cycle checkpoint loss", "significance": {"sensitivity": "CDK4/6 pathway context", "resistance": "None", "prognosis": "Poor"}}],
        "CDH1": [{"name": "E243K", "type": "missense", "pathogenicity": "Medium", "effect": "Adhesion loss", "significance": {"sensitivity": "Clinical trial options", "resistance": "None", "prognosis": "Intermediate"}}],
        "CHEK2": [{"name": "1100delC", "type": "frameshift", "pathogenicity": "High", "effect": "DNA-damage checkpoint loss", "significance": {"sensitivity": "PARP context dependent", "resistance": "None", "prognosis": "Intermediate"}}],
        "PALB2": [{"name": "Q775*", "type": "nonsense", "pathogenicity": "High", "effect": "Homologous recombination deficiency", "significance": {"sensitivity": "PARP inhibitors", "resistance": "Reversion variants", "prognosis": "Favorable with targeted therapy"}}],
        "RAD51C": [{"name": "R312W", "type": "missense", "pathogenicity": "Medium", "effect": "DNA repair dysfunction", "significance": {"sensitivity": "PARP context dependent", "resistance": "None", "prognosis": "Intermediate"}}],
        "RAD51D": [{"name": "K91fs", "type": "frameshift", "pathogenicity": "High", "effect": "DNA repair dysfunction", "significance": {"sensitivity": "PARP inhibitors", "resistance": "Reversion variants", "prognosis": "Intermediate"}}],
        "POLE": [{"name": "P286R", "type": "missense", "pathogenicity": "High", "effect": "Hypermutator phenotype", "significance": {"sensitivity": "Immunotherapy", "resistance": "None", "prognosis": "Variable"}}],
        "MLH1": [{"name": "E578*", "type": "nonsense", "pathogenicity": "High", "effect": "Mismatch repair loss", "significance": {"sensitivity": "Immunotherapy", "resistance": "None", "prognosis": "Variable"}}],
        "MSH2": [{"name": "A636P", "type": "missense", "pathogenicity": "High", "effect": "Mismatch repair loss", "significance": {"sensitivity": "Immunotherapy", "resistance": "None", "prognosis": "Variable"}}],
        "MSH6": [{"name": "F1088fs", "type": "frameshift", "pathogenicity": "High", "effect": "Mismatch repair loss", "significance": {"sensitivity": "Immunotherapy", "resistance": "None", "prognosis": "Variable"}}],
        "PMS2": [{"name": "R563*", "type": "nonsense", "pathogenicity": "Medium", "effect": "Mismatch repair attenuation", "significance": {"sensitivity": "Immunotherapy", "resistance": "None", "prognosis": "Intermediate"}}],
        "TSC1": [{"name": "Q527*", "type": "nonsense", "pathogenicity": "High", "effect": "mTOR pathway dysregulation", "significance": {"sensitivity": "mTOR inhibitors", "resistance": "PI3K bypass", "prognosis": "Intermediate"}}],
        "TSC2": [{"name": "R611Q", "type": "missense", "pathogenicity": "Medium", "effect": "mTOR pathway activation", "significance": {"sensitivity": "mTOR inhibitors", "resistance": "PI3K bypass", "prognosis": "Intermediate"}}],
        "MTOR": [{"name": "S2215Y", "type": "missense", "pathogenicity": "Medium", "effect": "Kinase activation", "significance": {"sensitivity": "mTOR inhibitors", "resistance": "Feedback signaling", "prognosis": "Variable"}}],
        "AKT1": [{"name": "E17K", "type": "missense", "pathogenicity": "High", "effect": "PI3K/AKT activation", "significance": {"sensitivity": "AKT inhibitors", "resistance": "RTK bypass", "prognosis": "Intermediate"}}],
        "PIK3R1": [{"name": "N564D", "type": "missense", "pathogenicity": "Medium", "effect": "PI3K regulation loss", "significance": {"sensitivity": "PI3K pathway therapies", "resistance": "None", "prognosis": "Variable"}}],
        "PTPN11": [{"name": "E76K", "type": "missense", "pathogenicity": "High", "effect": "RAS/MAPK activation", "significance": {"sensitivity": "MEK pathway approaches", "resistance": "RTK inhibitors", "prognosis": "Intermediate"}}],
        "MAP2K1": [{"name": "K57N", "type": "missense", "pathogenicity": "High", "effect": "MAPK activation", "significance": {"sensitivity": "MEK inhibitors", "resistance": "RAF inhibitors", "prognosis": "Intermediate"}}],
        "MAP2K2": [{"name": "Q60P", "type": "missense", "pathogenicity": "Medium", "effect": "MAPK activation", "significance": {"sensitivity": "MEK pathway approaches", "resistance": "RAF inhibitors", "prognosis": "Intermediate"}}],
        "RAF1": [{"name": "S257L", "type": "missense", "pathogenicity": "Medium", "effect": "MAPK activation", "significance": {"sensitivity": "MEK inhibitors", "resistance": "None", "prognosis": "Variable"}}],
        "FLT3": [{"name": "ITD", "type": "deletion", "pathogenicity": "High", "effect": "Constitutive signaling", "significance": {"sensitivity": "FLT3 inhibitors", "resistance": "Secondary FLT3 mutation", "prognosis": "Poor"}}],
        "NPM1": [{"name": "W288fs", "type": "frameshift", "pathogenicity": "High", "effect": "Aberrant localization", "significance": {"sensitivity": "Risk-stratified therapy", "resistance": "None", "prognosis": "Intermediate"}}],
        "RUNX1": [{"name": "R201Q", "type": "missense", "pathogenicity": "Medium", "effect": "Transcriptional dysregulation", "significance": {"sensitivity": "Clinical trial options", "resistance": "None", "prognosis": "Poor"}}],
        "DNMT3A": [{"name": "R882H", "type": "missense", "pathogenicity": "High", "effect": "Epigenetic dysregulation", "significance": {"sensitivity": "Hypomethylating agents", "resistance": "None", "prognosis": "Intermediate"}}],
        "TET2": [{"name": "Q548*", "type": "nonsense", "pathogenicity": "Medium", "effect": "Epigenetic dysregulation", "significance": {"sensitivity": "Hypomethylating agents", "resistance": "None", "prognosis": "Intermediate"}}],
        "SF3B1": [{"name": "K700E", "type": "missense", "pathogenicity": "Medium", "effect": "Spliceosome alteration", "significance": {"sensitivity": "Clinical trial options", "resistance": "None", "prognosis": "Variable"}}],
        "VHL": [{"name": "R167Q", "type": "missense", "pathogenicity": "High", "effect": "HIF stabilization", "significance": {"sensitivity": "VEGF/HIF axis therapies", "resistance": "None", "prognosis": "Intermediate"}}],
        "NF2": [{"name": "Q324*", "type": "nonsense", "pathogenicity": "High", "effect": "Hippo pathway loss", "significance": {"sensitivity": "FAK/mTOR context", "resistance": "None", "prognosis": "Intermediate"}}],
        "SMARCA4": [{"name": "R1192H", "type": "missense", "pathogenicity": "Medium", "effect": "Chromatin remodeling loss", "significance": {"sensitivity": "Clinical trial options", "resistance": "None", "prognosis": "Poor"}}],
        "SMARCB1": [{"name": "R377*", "type": "nonsense", "pathogenicity": "High", "effect": "SWI/SNF complex loss", "significance": {"sensitivity": "EZH2 inhibitors", "resistance": "None", "prognosis": "Poor"}}],
        "EZH2": [{"name": "Y646N", "type": "missense", "pathogenicity": "High", "effect": "Epigenetic gain-of-function", "significance": {"sensitivity": "EZH2 inhibitors", "resistance": "Secondary EZH2 mutation", "prognosis": "Variable"}}],
        "NOTCH1": [{"name": "P2514fs", "type": "frameshift", "pathogenicity": "Medium", "effect": "Notch signaling alteration", "significance": {"sensitivity": "Gamma-secretase context", "resistance": "None", "prognosis": "Variable"}}],
        "SMO": [{"name": "W535L", "type": "missense", "pathogenicity": "High", "effect": "Hedgehog activation", "significance": {"sensitivity": "SMO inhibitors", "resistance": "SMO mutations", "prognosis": "Intermediate"}}],
        "TERT": [{"name": "C228T", "type": "missense", "pathogenicity": "Medium", "effect": "Promoter activation", "significance": {"sensitivity": "None", "resistance": "None", "prognosis": "Poor"}}],
        "GNAQ": [{"name": "Q209L", "type": "missense", "pathogenicity": "High", "effect": "MAPK activation", "significance": {"sensitivity": "PKC/MEK context", "resistance": "None", "prognosis": "Intermediate"}}],
        "GNA11": [{"name": "Q209P", "type": "missense", "pathogenicity": "High", "effect": "MAPK activation", "significance": {"sensitivity": "PKC/MEK context", "resistance": "None", "prognosis": "Intermediate"}}],
        "AR": [{"name": "T878A", "type": "missense", "pathogenicity": "High", "effect": "Ligand promiscuity", "significance": {"sensitivity": "AR pathway inhibitors", "resistance": "Castration resistance", "prognosis": "Intermediate"}}],
        "FOXA1": [{"name": "H247Y", "type": "missense", "pathogenicity": "Medium", "effect": "Hormone signaling reprogramming", "significance": {"sensitivity": "Endocrine context", "resistance": "None", "prognosis": "Variable"}}],
        "PDCD1LG2": [{"name": "Amplification", "type": "amplification", "pathogenicity": "Medium", "effect": "Immune checkpoint activation", "significance": {"sensitivity": "Immunotherapy", "resistance": "Primary immune resistance", "prognosis": "Variable"}}],
        "CD274": [{"name": "Amplification", "type": "amplification", "pathogenicity": "Medium", "effect": "PD-L1 overexpression", "significance": {"sensitivity": "Immunotherapy", "resistance": "Primary immune resistance", "prognosis": "Variable"}}],
    }
)


@st.cache_data
def generate_mutations():
    rng = random.Random(RNG_SEED)
    dataset = []
    effects_pool = ["Loss of function", "Pathway activation", "Dominant negative", "Gain of function"]
    sensitivities = ["Targeted therapies", "Immunotherapy", "Standard chemotherapy", "Investigational"]
    resistances = ["TKIs", "Chemotherapy", "Endocrine therapy", "Pathway bypass", "None"]
    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

    for gene, variants in KNOWN_VARIANTS.items():
        for variant in variants:
            copied = {
                "gene": gene,
                "name": variant["name"],
                "type": variant["type"],
                "pathogenicity": variant["pathogenicity"],
                "effect": variant["effect"],
                "significance": {
                    "sensitivity": variant["significance"]["sensitivity"],
                    "resistance": variant["significance"]["resistance"],
                    "prognosis": variant["significance"]["prognosis"],
                },
            }
            refs, evidence = build_pubmed_refs(gene, variant["name"], known=True)
            copied["tier"] = infer_tier(copied["pathogenicity"], copied["significance"]["sensitivity"])
            copied["classification"] = infer_classification(copied["pathogenicity"], copied["significance"]["sensitivity"])
            copied["pubmed_articles"] = refs
            copied["evidence_type"] = evidence
            copied["source_hint"] = GENE_SOURCE_MAP.get(gene, "ClinVar/PubMed-informed oncology panel")
            dataset.append(copied)

    for gene in GENES:
        existing_names = {entry["name"] for entry in dataset if entry["gene"] == gene}
        target_count = rng.randint(6, 10)

        while len(existing_names) < target_count:
            mut_type = rng.choice(MUTATION_TYPES)
            patho = rng.choices(PATHOGENICITY, weights=[0.45, 0.35, 0.20], k=1)[0]

            if mut_type == "missense":
                name = f"{rng.choice(amino_acids)}{rng.randint(20, 950)}{rng.choice(amino_acids)}"
            elif mut_type == "nonsense":
                name = f"{rng.choice(amino_acids)}{rng.randint(20, 950)}*"
            elif mut_type == "frameshift":
                name = f"{rng.choice(amino_acids)}{rng.randint(20, 950)}fs"
            elif mut_type == "deletion":
                start = rng.randint(20, 700)
                name = f"del{start}_{start + rng.randint(1, 12)}"
            elif mut_type == "amplification":
                name = "Amplification"
            else:
                partner = rng.choice(["EML4", "KIF5B", "CD74", "TPM3", "CCDC6", "NCOA4"])
                name = f"{partner}-{gene} Fusion"

            if name in existing_names:
                continue

            dataset.append(
                {
                    "gene": gene,
                    "name": name,
                    "type": mut_type,
                    "pathogenicity": patho,
                    "effect": rng.choice(effects_pool),
                    "significance": {
                        "sensitivity": rng.choice(sensitivities) if patho == "High" else "None",
                        "resistance": rng.choice(resistances) if rng.random() > 0.50 else "None",
                        "prognosis": rng.choice(["Poor", "Intermediate", "Good", "Unknown"]),
                    },
                }
            )
            dataset[-1]["tier"] = infer_tier(dataset[-1]["pathogenicity"], dataset[-1]["significance"]["sensitivity"])
            dataset[-1]["classification"] = infer_classification(dataset[-1]["pathogenicity"], dataset[-1]["significance"]["sensitivity"])
            refs, evidence = build_pubmed_refs(gene, name, known=False)
            dataset[-1]["pubmed_articles"] = refs
            dataset[-1]["evidence_type"] = evidence
            dataset[-1]["source_hint"] = GENE_SOURCE_MAP.get(gene, "ClinVar/PubMed-informed oncology panel")
            existing_names.add(name)
    return dataset


MUTATION_DB = generate_mutations()

# -----------------
# 2. Mock Drug Dataset
# -----------------
DRUG_DB = [
    {"name": "Olaparib", "target_genes": ["BRCA1", "BRCA2", "ATM"], "pathways": ["DNA Repair", "PARP"], "toxicity": 0.4, "cost": 0.8, "resistance_markers": ["BRCA1-Rev"]},
    {"name": "Sotorasib", "target_genes": ["KRAS"], "pathways": ["MAPK"], "toxicity": 0.3, "cost": 0.9, "resistance_markers": ["Secondary KRAS mut", "EGFR"]},
    {"name": "Erlotinib", "target_genes": ["EGFR"], "pathways": ["RTK", "MAPK"], "toxicity": 0.5, "cost": 0.5, "resistance_markers": ["T790M", "C797S"]},
    {"name": "Osimertinib", "target_genes": ["EGFR"], "pathways": ["RTK", "MAPK"], "toxicity": 0.4, "cost": 0.9, "resistance_markers": ["C797S"]},
    {"name": "Pembrolizumab", "target_genes": ["PD-L1", "TP53"], "pathways": ["Immune System"], "toxicity": 0.2, "cost": 0.85, "resistance_markers": ["Low TMB", "STK11"]},
    {"name": "Vemurafenib", "target_genes": ["BRAF"], "pathways": ["MAPK"], "toxicity": 0.5, "cost": 0.6, "resistance_markers": ["NRAS", "MEK"]},
    {"name": "Trametinib", "target_genes": ["MEK1", "MEK2", "BRAF", "NF1"], "pathways": ["MAPK"], "toxicity": 0.6, "cost": 0.7, "resistance_markers": ["AKT amp"]},
    {"name": "Palbociclib", "target_genes": ["CDK4", "CDK6"], "pathways": ["Cell Cycle"], "toxicity": 0.45, "cost": 0.75, "resistance_markers": ["RB1 mut"]},
    {"name": "Trastuzumab", "target_genes": ["ERBB2", "ERBB3"], "pathways": ["RTK", "PI3K"], "toxicity": 0.3, "cost": 0.8, "resistance_markers": ["PTEN loss"]},
    {"name": "Crizotinib", "target_genes": ["ALK", "ROS1", "MET"], "pathways": ["RTK"], "toxicity": 0.4, "cost": 0.85, "resistance_markers": ["L1196M", "G2032R"]},
    {"name": "Alpelisib", "target_genes": ["PIK3CA", "PTEN"], "pathways": ["PI3K"], "toxicity": 0.6, "cost": 0.78, "resistance_markers": ["PTEN loss"]},
    {"name": "Selpercatinib", "target_genes": ["RET"], "pathways": ["RTK"], "toxicity": 0.35, "cost": 0.88, "resistance_markers": ["RET solvent-front mutation"]},
    {"name": "Larotrectinib", "target_genes": ["NTRK1", "NTRK2", "NTRK3"], "pathways": ["RTK"], "toxicity": 0.25, "cost": 0.92, "resistance_markers": ["TRK kinase domain mutation"]},
    {"name": "Futibatinib", "target_genes": ["FGFR2", "FGFR3"], "pathways": ["RTK"], "toxicity": 0.45, "cost": 0.86, "resistance_markers": ["FGFR gatekeeper"]},
    {"name": "Niraparib", "target_genes": ["BRCA1", "BRCA2", "PALB2"], "pathways": ["DNA Repair", "PARP"], "toxicity": 0.46, "cost": 0.82, "resistance_markers": ["BRCA2 reversion", "PARP escape"]},
    {"name": "Talazoparib", "target_genes": ["BRCA1", "BRCA2", "RAD51D"], "pathways": ["DNA Repair", "PARP"], "toxicity": 0.5, "cost": 0.9, "resistance_markers": ["BRCA1 reversion"]},
    {"name": "Capivasertib", "target_genes": ["AKT1", "PIK3CA", "PTEN"], "pathways": ["PI3K"], "toxicity": 0.5, "cost": 0.76, "resistance_markers": ["RTK bypass", "MTOR"]},
    {"name": "Everolimus", "target_genes": ["MTOR", "TSC1", "TSC2"], "pathways": ["PI3K", "Cell Cycle"], "toxicity": 0.48, "cost": 0.56, "resistance_markers": ["PI3K bypass"]},
    {"name": "Temsirolimus", "target_genes": ["MTOR", "TSC1", "TSC2"], "pathways": ["PI3K"], "toxicity": 0.46, "cost": 0.54, "resistance_markers": ["AKT activation"]},
    {"name": "Dabrafenib", "target_genes": ["BRAF"], "pathways": ["MAPK"], "toxicity": 0.47, "cost": 0.74, "resistance_markers": ["MEK", "NRAS"]},
    {"name": "Encorafenib", "target_genes": ["BRAF"], "pathways": ["MAPK"], "toxicity": 0.42, "cost": 0.78, "resistance_markers": ["MAP2K1"]},
    {"name": "Binimetinib", "target_genes": ["MAP2K1", "MAP2K2", "NF1"], "pathways": ["MAPK"], "toxicity": 0.56, "cost": 0.7, "resistance_markers": ["RAF1"]},
    {"name": "Cobimetinib", "target_genes": ["MAP2K1", "MAP2K2"], "pathways": ["MAPK"], "toxicity": 0.58, "cost": 0.72, "resistance_markers": ["BRAF splice"]},
    {"name": "Abemaciclib", "target_genes": ["CDK4", "CDK6", "CCND1"], "pathways": ["Cell Cycle"], "toxicity": 0.52, "cost": 0.73, "resistance_markers": ["RB1 mut"]},
    {"name": "Ribociclib", "target_genes": ["CDK4", "CDK6", "CCND3"], "pathways": ["Cell Cycle"], "toxicity": 0.47, "cost": 0.71, "resistance_markers": ["CDK6 amplification"]},
    {"name": "Atezolizumab", "target_genes": ["CD274", "PD-L1"], "pathways": ["Immune System"], "toxicity": 0.24, "cost": 0.83, "resistance_markers": ["STK11", "Low TMB"]},
    {"name": "Nivolumab", "target_genes": ["PDCD1LG2", "PD-L1"], "pathways": ["Immune System"], "toxicity": 0.23, "cost": 0.84, "resistance_markers": ["Primary immune resistance"]},
    {"name": "Ipilimumab", "target_genes": ["CTLA4", "PD-L1"], "pathways": ["Immune System"], "toxicity": 0.62, "cost": 0.87, "resistance_markers": ["Immune exhaustion"]},
    {"name": "Alectinib", "target_genes": ["ALK"], "pathways": ["RTK", "MAPK"], "toxicity": 0.33, "cost": 0.88, "resistance_markers": ["G1202R"]},
    {"name": "Lorlatinib", "target_genes": ["ALK", "ROS1"], "pathways": ["RTK"], "toxicity": 0.4, "cost": 0.93, "resistance_markers": ["Compound ALK mutations"]},
    {"name": "Entrectinib", "target_genes": ["NTRK1", "NTRK2", "NTRK3", "ROS1"], "pathways": ["RTK"], "toxicity": 0.38, "cost": 0.9, "resistance_markers": ["TRK solvent-front"]},
    {"name": "Tepotinib", "target_genes": ["MET"], "pathways": ["RTK"], "toxicity": 0.36, "cost": 0.87, "resistance_markers": ["MET amplification escape"]},
    {"name": "Capmatinib", "target_genes": ["MET"], "pathways": ["RTK"], "toxicity": 0.37, "cost": 0.86, "resistance_markers": ["MET D1228N"]},
    {"name": "Pralsetinib", "target_genes": ["RET"], "pathways": ["RTK"], "toxicity": 0.34, "cost": 0.89, "resistance_markers": ["RET G810"]},
    {"name": "Erdafitinib", "target_genes": ["FGFR1", "FGFR2", "FGFR3"], "pathways": ["RTK"], "toxicity": 0.44, "cost": 0.85, "resistance_markers": ["FGFR gatekeeper"]},
    {"name": "Ponatinib", "target_genes": ["FGFR2", "PDGFRA", "FLT3"], "pathways": ["RTK", "MAPK"], "toxicity": 0.67, "cost": 0.86, "resistance_markers": ["Vascular toxicity limit"]},
    {"name": "Gilteritinib", "target_genes": ["FLT3"], "pathways": ["RTK"], "toxicity": 0.41, "cost": 0.81, "resistance_markers": ["FLT3 F691L"]},
    {"name": "Ivosidenib", "target_genes": ["IDH1"], "pathways": ["PI3K"], "toxicity": 0.29, "cost": 0.79, "resistance_markers": ["Isoform switching"]},
    {"name": "Enasidenib", "target_genes": ["IDH2"], "pathways": ["PI3K"], "toxicity": 0.31, "cost": 0.8, "resistance_markers": ["Secondary IDH2 mutation"]},
    {"name": "Ruxolitinib", "target_genes": ["JAK1", "JAK2"], "pathways": ["Immune System", "Cell Cycle"], "toxicity": 0.43, "cost": 0.69, "resistance_markers": ["JAK pathway reactivation"]},
    {"name": "Venetoclax", "target_genes": ["BCL2"], "pathways": ["Cell Cycle"], "toxicity": 0.45, "cost": 0.77, "resistance_markers": ["MCL1 upregulation"]},
    {"name": "Ibrutinib", "target_genes": ["BTK"], "pathways": ["Immune System"], "toxicity": 0.39, "cost": 0.74, "resistance_markers": ["BTK C481S"]},
]


# -----------------
# 3. Scoring Engine
# -----------------
def compute_score(drug_combo, variant, mutation_type, cancer_type):
    breakdown = {
        "mutation_impact": 0.0,
        "pathway_relevance": 0.0,
        "drug_sensitivity": 0.0,
        "resistance_penalty": 0.0,
    }

    all_targets = set()
    all_pathways = set()
    all_resistance = set()

    for drug in drug_combo:
        all_targets.update(drug["target_genes"])
        all_pathways.update(drug["pathways"])
        all_resistance.update(drug["resistance_markers"])

    relevant_pathways = CANCER_PATHWAY_MAP.get(cancer_type, [])

    if variant and variant["gene"] in all_targets:
        if variant["pathogenicity"] == "High":
            breakdown["mutation_impact"] = 3.0
        elif variant["pathogenicity"] == "Medium":
            breakdown["mutation_impact"] = 1.5
        else:
            breakdown["mutation_impact"] = 0.5

    if any(pathway in relevant_pathways for pathway in all_pathways):
        breakdown["pathway_relevance"] = 1.0

    if mutation_type == "somatic":
        breakdown["drug_sensitivity"] += 0.5

    if variant and variant["significance"]["sensitivity"] != "None":
        breakdown["drug_sensitivity"] += 1.0

    if len(drug_combo) == 2:
        breakdown["drug_sensitivity"] += 0.5

    if variant:
        for marker in all_resistance:
            marker_lower = marker.lower()
            if variant["name"].lower() in marker_lower or variant["gene"].lower() in marker_lower:
                breakdown["resistance_penalty"] = -2.0

    base_score = sum(breakdown.values())
    tox = max(drug["toxicity"] for drug in drug_combo)
    cost = min(1.0, sum(drug["cost"] for drug in drug_combo))

    return base_score, breakdown, tox, cost


# -----------------
# 4. Generator & Optimizations
# -----------------
def generate_combinations(variant, mutation_type, cancer_type, strategy="Both", min_raw_score=0.0):
    combos = []
    include_single = strategy in ["Both", "Single Drug"]
    include_pairs = strategy in ["Both", "Combination"]

    if include_single:
        for drug in DRUG_DB:
            score, breakdown, tox, cost = compute_score([drug], variant, mutation_type, cancer_type)
            if score < min_raw_score:
                continue
            combos.append(
                {
                    "drugs": [drug],
                    "drug_names": [drug["name"]],
                    "raw_score": score,
                    "breakdown": breakdown,
                    "toxicity": tox,
                    "cost": cost,
                    "combo_type": "1-drug",
                    "combo_key": tuple(sorted([drug["name"]])),
                }
            )

    if include_pairs:
        for pair in itertools.combinations(DRUG_DB, 2):
            score, breakdown, tox, cost = compute_score(list(pair), variant, mutation_type, cancer_type)
            if score < min_raw_score:
                continue
            names = [drug["name"] for drug in pair]
            combos.append(
                {
                    "drugs": list(pair),
                    "drug_names": names,
                    "raw_score": score,
                    "breakdown": breakdown,
                    "toxicity": tox,
                    "cost": cost,
                    "combo_type": "2-drug",
                    "combo_key": tuple(sorted(names)),
                }
            )

    return combos


def pareto_filter(combos, w_eff, w_tox, w_cost):
    pareto_front = []
    for i, c1 in enumerate(combos):
        dominated = False
        for j, c2 in enumerate(combos):
            if i == j:
                continue

            score_ge = c2["raw_score"] >= c1["raw_score"]
            tox_le = c2["toxicity"] <= c1["toxicity"]
            cost_le = c2["cost"] <= c1["cost"]

            score_gt = c2["raw_score"] > c1["raw_score"]
            tox_lt = c2["toxicity"] < c1["toxicity"]
            cost_lt = c2["cost"] < c1["cost"]

            if (score_ge and tox_le and cost_le) and (score_gt or tox_lt or cost_lt):
                dominated = True
                break

        if not dominated:
            weighted = (c1["raw_score"] * w_eff) - (c1["toxicity"] * w_tox * 3) - (c1["cost"] * w_cost * 3)
            c1_copy = dict(c1)
            c1_copy["weighted_score"] = weighted
            pareto_front.append(c1_copy)

    pareto_front.sort(
        key=lambda item: (
            item["weighted_score"],
            item["raw_score"],
            -item["toxicity"],
            -item["cost"],
            " + ".join(item["drug_names"]),
        ),
        reverse=True,
    )
    return pareto_front


def get_counterfactuals(combo, original_variant, mutation_type, cancer_type, w_eff, w_tox, w_cost):
    score_no_mut, _, tox_no_mut, cost_no_mut = compute_score(combo["drugs"], None, mutation_type, cancer_type)
    weighted_no_mut = (score_no_mut * w_eff) - (tox_no_mut * w_tox * 3) - (cost_no_mut * w_cost * 3)

    drops = []
    if len(combo["drugs"]) == 2:
        first, second = combo["drugs"]
        score_1, _, tox_1, cost_1 = compute_score([first], original_variant, mutation_type, cancer_type)
        weighted_1 = (score_1 * w_eff) - (tox_1 * w_tox * 3) - (cost_1 * w_cost * 3)

        score_2, _, tox_2, cost_2 = compute_score([second], original_variant, mutation_type, cancer_type)
        weighted_2 = (score_2 * w_eff) - (tox_2 * w_tox * 3) - (cost_2 * w_cost * 3)

        drops.append({"dropped": second["name"], "new_w_score": weighted_1, "delta": weighted_1 - combo["weighted_score"]})
        drops.append({"dropped": first["name"], "new_w_score": weighted_2, "delta": weighted_2 - combo["weighted_score"]})

    return {
        "score_no_mut": weighted_no_mut,
        "delta_no_mut": weighted_no_mut - combo["weighted_score"],
        "drops": drops,
    }


def filter_gene_variants(gene, text_query="", pathogenicity_filter=None, mutation_filter=None):
    filtered = [item for item in MUTATION_DB if item["gene"] == gene]

    if text_query:
        query = text_query.lower().strip()
        filtered = [item for item in filtered if query in item["name"].lower() or query in item["effect"].lower()]

    if pathogenicity_filter and "All" not in pathogenicity_filter:
        filtered = [item for item in filtered if item["pathogenicity"] in pathogenicity_filter]

    if mutation_filter and "All" not in mutation_filter:
        filtered = [item for item in filtered if item["type"] in mutation_filter]

    return filtered


def pick_scoring_variant(selected_variant, visible_variants):
    if selected_variant:
        return selected_variant, False
    if not visible_variants:
        return None, False

    patho_rank = {"High": 0, "Medium": 1, "Low": 2}
    tier_rank = {"Tier I": 0, "Tier II": 1, "Tier III": 2, "Tier IV": 3}
    ranked = sorted(
        visible_variants,
        key=lambda item: (
            patho_rank.get(item.get("pathogenicity", "Low"), 3),
            tier_rank.get(item.get("tier", "Tier IV"), 4),
            item.get("name", ""),
        ),
    )
    return ranked[0], True


def recommendation_rationale(combo):
    benefit = "High pathway match with balanced toxicity." if combo["toxicity"] <= 0.45 else "Strong efficacy contribution from pathway-aligned targets."
    risk = "Higher expected adverse-event burden." if combo["toxicity"] > 0.55 else "Moderate safety profile; monitor tolerability."
    caveat = "Potential resistance signal requires molecular re-evaluation on progression." if combo["breakdown"]["resistance_penalty"] < 0 else "No explicit resistance flag in this simulation cycle."
    why = "Selected for strong weighted performance under your objective settings and Pareto optimality."
    return why, benefit, risk, caveat


def regimen_scientific_details(combo, variant, cancer_type):
    all_targets = sorted({target for drug in combo["drugs"] for target in drug["target_genes"]})
    all_pathways = sorted({pathway for drug in combo["drugs"] for pathway in drug["pathways"]})
    resistance_markers = sorted({marker for drug in combo["drugs"] for marker in drug["resistance_markers"]})
    relevant_pathways = set(CANCER_PATHWAY_MAP.get(cancer_type, []))
    pathway_overlap = [pathway for pathway in all_pathways if pathway in relevant_pathways]

    if pathway_overlap:
        mechanism_line = (
            f"Mechanistic rationale: regimen engages {', '.join(all_pathways)}; "
            f"cancer-context pathway overlap is {', '.join(pathway_overlap)}."
        )
    else:
        mechanism_line = (
            f"Mechanistic rationale: regimen engages {', '.join(all_pathways)}; "
            "no direct canonical pathway overlap is detected for this tumor context."
        )

    if variant:
        direct_target = variant["gene"] in all_targets
        sensitivity_signal = variant["significance"].get("sensitivity", "None")
        if sensitivity_signal != "None":
            sensitivity_line = (
                f"Sensitivity context: {variant['gene']} {variant['name']} has reported sensitivity to {sensitivity_signal}; "
                f"direct target engagement in this regimen is {'present' if direct_target else 'not explicit'} "
                f"(targets: {', '.join(all_targets)})."
            )
        else:
            sensitivity_line = (
                f"Sensitivity context: {variant['gene']} {variant['name']} has no explicit sensitivity label; "
                f"prioritization is driven by mutation impact/pathway relevance with {'present' if direct_target else 'indirect'} target linkage."
            )

        marker_hits = [m for m in resistance_markers if variant["name"].lower() in m.lower() or variant["gene"].lower() in m.lower()]
        variant_resistance = variant["significance"].get("resistance", "None")
        if marker_hits:
            resistance_line = (
                f"Resistance assessment: overlap with regimen resistance markers ({', '.join(marker_hits)}); "
                "this contributes a modeled resistance penalty."
            )
        elif variant_resistance != "None":
            resistance_line = (
                f"Resistance assessment: variant-level resistance signal is '{variant_resistance}', "
                "but no direct marker overlap is detected in the selected drug set."
            )
        else:
            resistance_line = "Resistance assessment: no explicit resistance signal detected for this variant-regimen pairing."
    else:
        direct_target = False
        marker_hits = []
        sensitivity_line = (
            f"Sensitivity context: gene-level evaluation without a specific variant; "
            f"actionability inferred from pathway alignment and target coverage ({', '.join(all_targets)})."
        )
        resistance_line = "Resistance assessment: variant-specific resistance could not be computed in gene-only mode."

    drug_lines = [
        f"{drug['name']}: targets {', '.join(drug['target_genes'])}; pathways {', '.join(drug['pathways'])}; resistance markers {', '.join(drug['resistance_markers'])}"
        for drug in combo["drugs"]
    ]

    return {
        "mechanism_line": mechanism_line,
        "sensitivity_line": sensitivity_line,
        "resistance_line": resistance_line,
        "drug_lines": drug_lines,
        "pathway_overlap": pathway_overlap,
        "direct_target": direct_target,
        "marker_hits": marker_hits,
    }


def pubmed_links_from_text(pubmed_text):
    if not pubmed_text:
        return []
    links = []
    chunks = [chunk.strip() for chunk in pubmed_text.split(";") if chunk.strip()]
    for chunk in chunks:
        pmid = chunk.replace("PMID:", "").strip()
        if pmid.isdigit():
            links.append((pmid, f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
    return links


def pubmed_markdown(pubmed_text):
    links = pubmed_links_from_text(pubmed_text)
    if not links:
        return "N/A"
    return "  ".join([f"[PMID:{pmid}]({url})" for pmid, url in links])


def explain_score_model():
    st.markdown(
        """
        **How to read model outputs**
        - **Clinical Benefit (Efficacy)**: combines mutation impact, pathway fit, and sensitivity signals.
        - **Safety Burden (Toxicity)**: higher values imply greater expected side-effect load.
        - **Economic Burden (Cost)**: normalized estimate of treatment expense pressure.
        - **Resistance Signal**: negative component when variant context overlaps known resistance markers.
        """
    )


WEIGHT_PRESETS = {
    "Balanced": {"eff": 0.8, "tox": 0.5, "cost": 0.4},
    "Efficacy-first": {"eff": 1.0, "tox": 0.25, "cost": 0.2},
    "Safety-first": {"eff": 0.65, "tox": 0.95, "cost": 0.35},
    "Cost-aware": {"eff": 0.7, "tox": 0.45, "cost": 0.95},
}


def strategy_label_to_mode(label):
    return {
        "Any": "Both",
        "Single only": "Single Drug",
        "Combination only": "Combination",
    }.get(label, "Both")


# -----------------
# 5. UI Integration
# -----------------
def main():
    st.set_page_config(page_title="Precision Oncology AI", layout="wide", page_icon="🧬")
    if PLOTLY_IMPORT_ERROR is not None:
        st.error(
            "Missing dependency: plotly. Install required packages with `pip install -r requirements.txt` and restart the app."
        )
        st.stop()
    st.markdown(
        """
        <style>
            :root {
                --bg-soft: #f4f7fa;
                --bg-neutral: #eef3f7;
                --hero-start: #162b39;
                --hero-mid: #1e4f63;
                --hero-end: #2b6779;
                --panel-bg: #f9fcff;
                --card-border: #d0dbe3;
                --border-strong: #9fb3c2;
                --text-main: #1f2d3a;
                --text-subtle: #4d6173;
                --accent: #2f7f7b;
                --accent-2: #2f6996;
                --accent-3: #bf8a3e;
                --focus: #2f7f7b;
                --focus-ring: rgba(47, 127, 123, 0.2);
                --good: #2d7a6d;
                --warn: #9a6a2a;
                --risk: #a65757;
                --slate: #5d6d7b;
                --amber: #af7f3c;
                --rose: #b76472;
                --grad-card: linear-gradient(140deg, #ffffff 0%, #f5f9fc 100%);
                --grad-pill-main: linear-gradient(180deg, #e8f7f5 0%, #f5fcfb 100%);
                --grad-pill-risk: linear-gradient(180deg, #fef3e6 0%, #fffbf5 100%);
                --shadow-sm: 0 2px 8px rgba(15, 39, 63, 0.06);
                --shadow-md: 0 8px 18px rgba(15, 39, 63, 0.11);
            }
            .hero { border: 1px solid #2e6173; background: linear-gradient(135deg, var(--hero-start) 0%, var(--hero-mid) 54%, var(--hero-end) 100%); border-radius: 14px; padding: 16px; margin-bottom: 12px; color: #f8fbff; box-shadow: var(--shadow-sm); }
            .left-panel { border: 1px solid var(--card-border); background: var(--panel-bg); border-radius: 12px; padding: 12px; box-shadow: var(--shadow-sm); }
            .rationale-card { border: 1px solid var(--card-border); background: #ffffff; border-radius: 12px; padding: 12px; margin-bottom: 8px; color: var(--text-main); box-shadow: var(--shadow-sm); font-size: 0.9rem; line-height: 1.45; }
            .disclaimer { border-left: 3px solid #7393ab; background: var(--bg-soft); padding: 8px; border-radius: 6px; color: var(--text-main); font-size: 0.86rem; }
            .bubble-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
            .bubble { border-radius: 999px; padding: 4px 10px; font-size: 11px; font-weight: 700; border: 1px solid #ccd7df; background: #f9fbfd; color: var(--slate); cursor: pointer; transition: transform 140ms ease, border-color 180ms ease, background-color 180ms ease, box-shadow 180ms ease; }
            .bubble:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(33, 62, 88, 0.12); border-color: var(--border-strong); }
            .bubble-good { border-color: #b4ddd2; background: #eff9f5; color: var(--good); }
            .bubble-warn { border-color: #e5d5b3; background: #fdf8ef; color: var(--amber); }
            .bubble-risk { border-color: #e7c6cc; background: #fdf3f5; color: var(--rose); }
            .rec-title { display: flex; align-items: center; gap: 8px; }
            .rank-dot { width: 24px; height: 24px; border-radius: 50%; background: linear-gradient(180deg, #2f9d97 0%, #2f7f7b 100%); color: #ffffff; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; box-shadow: 0 3px 8px rgba(19, 72, 68, 0.24); }
            .chip-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0 10px 0; }
            .chip { border-radius: 999px; border: 1px solid #d3dce2; padding: 2px 8px; font-size: 10px; color: #415364; background: #f6f9fb; transition: background-color 180ms ease, border-color 180ms ease; }
            .chip:hover { background: #edf4f9; border-color: #b8c9d6; }
            .query-strip { border: 1px solid #d6e2ea; background: linear-gradient(180deg, #fbfdff 0%, #f2f8fd 100%); border-radius: 10px; padding: 7px 9px; opacity: 0.95; margin: 6px 0 10px 0; box-shadow: var(--shadow-sm); }
            .query-strip .chip-row { margin: 0; }
            .score-strip { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin: 6px 0; }
            .score-pill { border: 1px solid #d7e0e7; border-radius: 10px; padding: 7px 8px; background: #f9fbfd; position: relative; transition: box-shadow 180ms ease, transform 140ms ease; }
            .score-pill:hover { transform: translateY(-1px); box-shadow: 0 6px 12px rgba(16, 42, 67, 0.09); }
            .score-pill-main { border-color: #96d0cc; background: var(--grad-pill-main); }
            .score-pill-risk { border-color: #e2c79c; background: var(--grad-pill-risk); }
            .score-pill-main::before, .score-pill-risk::before { position: absolute; top: 6px; right: 8px; font-size: 0.72rem; font-weight: 800; opacity: 0.5; }
            .score-pill-main::before { content: "+"; color: #2f7f7b; }
            .score-pill-risk::before { content: "!"; color: #9a6a2a; }
            .score-label { font-size: 0.68rem; color: #5c6f80; line-height: 1.1; }
            .score-value { font-size: 0.94rem; color: #223544; font-weight: 800; line-height: 1.2; margin-top: 1px; }
            .rec-tags { display: flex; gap: 6px; flex-wrap: wrap; margin: 5px 0 4px 0; }
            .rec-tag { border-radius: 999px; padding: 2px 9px; font-size: 0.68rem; font-weight: 700; border: 1px solid #d3dce2; cursor: pointer; transition: transform 140ms ease, box-shadow 180ms ease; }
            .rec-tag:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(22, 46, 69, 0.12); }
            .tag-optimal { color: #1f6d68; border-color: #9fd4cf; background: #edf9f7; }
            .tag-balanced { color: #70531f; border-color: #e0c58e; background: #fdf6e8; }
            .tag-watch { color: #7f3f54; border-color: #e7c1cf; background: #fcf1f5; }
            .section-bar { margin: 4px 0 10px 0; padding: 6px 9px; border-radius: 8px; font-size: 0.78rem; font-weight: 700; border: 1px solid #d6dee4; letter-spacing: 0.02em; }
            .section-mut { background: #eef6f4; color: #2f665f; }
            .section-rec { background: #f5f6f8; color: #495769; }
            .section-viz { background: #f9f5ee; color: #7d6238; }
            .hint-line { color: var(--text-subtle); font-size: 0.76rem; margin: 2px 0 8px 0; }
            .rec-shell { border: 1px solid var(--card-border); border-radius: 14px; padding: 10px; margin: 0 0 12px 0; background: var(--grad-card); box-shadow: var(--shadow-sm); transition: transform 180ms ease, box-shadow 200ms ease, border-color 180ms ease; }
            .rec-shell:hover { transform: translateY(-1px); box-shadow: var(--shadow-md); border-color: #b5c8d7; }
            .rec-shell-top { border-color: #8bbec8; box-shadow: 0 10px 20px rgba(27, 88, 102, 0.15); }
            .rationale-grid { display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 6px; }
            .rationale-callout { border: 1px solid #d8e2e9; background: #fdfefe; border-radius: 10px; padding: 8px 9px; color: var(--text-main); }
            .rationale-k { display: block; font-size: 0.73rem; font-weight: 700; color: #335266; margin-bottom: 2px; letter-spacing: 0.01em; }
            .rationale-v { font-size: 0.84rem; line-height: 1.42; }
            .stButton > button, .stDownloadButton > button { transition: transform 140ms ease, box-shadow 180ms ease, border-color 180ms ease; }
            .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 14px rgba(18, 48, 74, 0.12); }
            .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--focus); box-shadow: 0 0 0 4px var(--focus-ring); }
            details[open] > summary { background: #eef5fa; border-radius: 8px; }
            @media (max-width: 900px) {
                .score-strip { grid-template-columns: 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hero'><h2 style='margin: 0; color: #f8fbff; font-size: 1.35rem;'>Precision Oncology Decision Simulator</h2><p style='margin: 6px 0 0 0; color: #dce8f7; font-size: 0.9rem;'>Guided demo workflow: setup genomic context, optimize treatment strategy, inspect tradeoffs, and explain rationale.</p></div>", unsafe_allow_html=True)

    total_known = sum(len(variants) for variants in KNOWN_VARIANTS.values())
    total_variants = len(MUTATION_DB)
    known_ratio = (total_known / total_variants) if total_variants else 0
    cols = st.columns(4)
    cols[0].metric("Genes in Catalog", len(GENES))
    cols[1].metric("Total Variants", total_variants)
    cols[2].metric("Known Variant Share", f"{known_ratio:.0%}")
    cols[3].metric("Therapy Options", len(DRUG_DB))

    if "filters_visible" not in st.session_state:
        st.session_state["filters_visible"] = True

    results = st.session_state.get("pipeline_results")

    action_left, action_mid, _ = st.columns([0.18, 0.18, 0.64])
    with action_left:
        if not st.session_state["filters_visible"]:
            if st.button("Show Filters", use_container_width=True):
                st.session_state["filters_visible"] = True
                st.rerun()
    with action_mid:
        if not st.session_state["filters_visible"] and results:
            if st.button("Edit Search", use_container_width=True):
                st.session_state["filters_visible"] = True
                st.rerun()

    if not st.session_state["filters_visible"] and results:
        st.markdown(
            f"<div class='query-strip'><div class='chip-row'><span class='chip'>Gene: {results['gene']}</span><span class='chip'>Variant (scored): {results.get('scoring_variant_name', 'None (gene-only search)')}</span><span class='chip'>Mode: {results.get('combo_mode_label', 'Any')}</span><span class='chip'>Cancer: {results['cancer_type']}</span></div></div>",
            unsafe_allow_html=True,
        )

    if st.session_state["filters_visible"]:
        layout_left, layout_right = st.columns([0.95, 2.05], gap="large")
    else:
        layout_left = st.container()
        layout_right = st.container()

    if st.session_state["filters_visible"]:
        with layout_left:
            st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
            st.markdown("### Filters")
            st.caption("Configure once, then press Enter or Go to apply across all tabs.")
            st.markdown("<div class='hint-line'>Hint: set tumor context first, then narrow variants only if you need precision filtering.</div>", unsafe_allow_html=True)
            with st.form("global_search_form"):
                with st.expander("Open Filters", expanded=False):
                    with st.expander("Tumor Context", expanded=True):
                        gene_search = st.text_input("Gene search", placeholder="Type gene symbol", key="gene_search")
                        filtered_genes = [g for g in GENES if gene_search.lower() in g.lower()] if gene_search else GENES
                        if not filtered_genes:
                            filtered_genes = GENES
                        gene = st.selectbox("Primary gene", filtered_genes, key="gene")
                        all_gene_variants = [item for item in MUTATION_DB if item["gene"] == gene]
                        c1, c2 = st.columns(2)
                        mutation_type = c1.radio("Origin", ["somatic", "germline"], horizontal=True, key="mutation_type")
                        cancer_type = c2.selectbox("Cancer indication", CANCER_TYPES, key="cancer_type")

                    with st.expander("Variant Filters", expanded=False):
                        variant_text_filter = st.text_input("Variant text filter", placeholder="e.g., V600E or fusion", key="variant_text_filter")
                        variant_patho_filter = st.multiselect("Pathogenicity", ["All", "High", "Medium", "Low"], default=["All"], key="variant_patho_filter")
                        variant_type_filter = st.multiselect("Mutation class", ["All"] + MUTATION_TYPES, default=["All"], key="variant_type_filter")
                        visible_variants = filter_gene_variants(gene, variant_text_filter, variant_patho_filter, variant_type_filter)
                        if not visible_variants:
                            visible_variants = all_gene_variants
                        variant_options = ["None (gene-only search)"] + [v["name"] for v in visible_variants]
                        selected_variant_name = st.selectbox("Candidate variant (optional)", variant_options, key="selected_variant_name")
                        variant = None if selected_variant_name == "None (gene-only search)" else next((v for v in visible_variants if v["name"] == selected_variant_name), None)

                    with st.expander("Optimization Controls", expanded=False):
                        preset = st.selectbox("Ranking preset", list(WEIGHT_PRESETS.keys()), index=0, key="preset")
                        d = WEIGHT_PRESETS[preset]
                        combo_mode_label = st.radio("Combo mode", ["Any", "Single only", "Combination only"], index=0, key="combo_mode_label")
                        top_n = st.slider("Top recommendations", 1, 10, 5, key="top_n")
                        min_raw_score = st.slider("Minimum score threshold", 0.0, 4.0, 0.0, 0.1, key="min_raw_score")
                        w_eff = st.slider("Efficacy weight", 0.0, 1.0, d["eff"], 0.05, key="w_eff")
                        w_tox = st.slider("Safety weight", 0.0, 1.0, d["tox"], 0.05, key="w_tox")
                        w_cost = st.slider("Cost weight", 0.0, 1.0, d["cost"], 0.05, key="w_cost")

                run_search = st.form_submit_button("Go", type="primary", use_container_width=True)

            st.markdown(
                f"<div class='chip-row'><span class='chip'>Gene: {gene}</span><span class='chip'>Variant: {selected_variant_name}</span><span class='chip'>Mode: {combo_mode_label}</span></div>",
                unsafe_allow_html=True,
            )

            if run_search:
                with st.spinner("Searching ranked therapy options..."):
                    scoring_variant, auto_selected_variant = pick_scoring_variant(variant, visible_variants)
                    strategy = strategy_label_to_mode(combo_mode_label)
                    all_combos = generate_combinations(
                        scoring_variant,
                        mutation_type,
                        cancer_type,
                        strategy=strategy,
                        min_raw_score=0.0,
                    )
                    pareto_combos = pareto_filter(all_combos, w_eff, w_tox, w_cost) if all_combos else []
                    weighted_all = []
                    optimal_keys = {combo["combo_key"] for combo in pareto_combos}
                    for combo in all_combos:
                        weighted = (combo["raw_score"] * w_eff) - (combo["toxicity"] * w_tox * 3) - (combo["cost"] * w_cost * 3)
                        combo_copy = dict(combo)
                        combo_copy["weighted_score"] = weighted
                        combo_copy["is_optimal"] = combo_copy["combo_key"] in optimal_keys
                        weighted_all.append(combo_copy)
                    weighted_all.sort(key=lambda item: item["weighted_score"], reverse=True)
                    time.sleep(0.2)
                st.session_state["pipeline_results"] = {
                    "all_combos": weighted_all,
                    "pareto": pareto_combos,
                    "variant": scoring_variant,
                    "selected_variant_name": selected_variant_name,
                    "scoring_variant_name": scoring_variant["name"] if scoring_variant else "None (gene-only search)",
                    "auto_selected_variant": auto_selected_variant,
                    "gene": gene,
                    "visible_variants": visible_variants,
                    "mutation_type": mutation_type,
                    "cancer_type": cancer_type,
                    "weights": {"eff": w_eff, "tox": w_tox, "cost": w_cost},
                    "top_n": top_n,
                    "combo_mode_label": combo_mode_label,
                }
                st.session_state["filters_visible"] = False
                st.toast("Search completed. Filters applied across all tabs.", icon="✅")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with layout_right:
        if not st.session_state.get("pipeline_results"):
            st.info("Run Search Recommendations in Patient Setup to view results.")
            explain_score_model()
            return
        results = st.session_state.get("pipeline_results")
        if not results:
            st.info("No results available yet.")
            return

        pareto_combos = results["pareto"]
        all_combos = results["all_combos"]
        gene = results["gene"]
        gene_variants = results["visible_variants"]
        variant = results["variant"]
        top_n = min(results["top_n"], max(1, len(pareto_combos))) if pareto_combos else 0
        w_eff = results["weights"]["eff"]
        w_tox = results["weights"]["tox"]
        w_cost = results["weights"]["cost"]

        explain_score_model()
        st.caption(
            f"Weights in effect -> Efficacy: {w_eff:.2f}, Safety: {w_tox:.2f}, Cost: {w_cost:.2f} | "
            f"Scoring mutation: {results.get('scoring_variant_name', 'None (gene-only search)')}"
        )
        st.markdown("<div class='disclaimer'>This app is a simulation for education and project demonstration. It is not clinical advice and must not be used for treatment decisions.</div>", unsafe_allow_html=True)

        if not pareto_combos:
            st.warning("No treatment options passed current filters. Lower the threshold or broaden combo mode.")
            return

        pane_mut, pane_reco, pane_viz = st.tabs(["Mutation Insights", "Recommendations", "Visual Analytics"])

        with pane_mut:
            st.markdown("<div class='section-bar section-mut'>Mutation Insights</div>", unsafe_allow_html=True)
            q1, q3, q4 = st.columns([1.2, 1.0, 0.8])
            mut_search = q1.text_input("Quick search", placeholder="Variant name or effect")
            sort_by = q3.selectbox("Sort by", ["tier", "pathogenicity", "name", "evidence"], index=0)
            asc_sort = q4.toggle("Ascending", value=True)

            tier_rank = {"Tier I": 1, "Tier II": 2, "Tier III": 3, "Tier IV": 4}
            patho_rank = {"High": 1, "Medium": 2, "Low": 3}
            evidence_order = {"Experimental evidence": 1, "Literature evidence": 2}

            filtered_mutations = gene_variants
            if mut_search:
                q = mut_search.lower().strip()
                filtered_mutations = [
                    item for item in filtered_mutations if q in item["name"].lower() or q in item["effect"].lower()
                ]
            if sort_by == "tier":
                filtered_mutations = sorted(
                    filtered_mutations,
                    key=lambda item: tier_rank.get(item["tier"] if isinstance(item, dict) and "tier" in item else "Tier IV", 5),
                    reverse=not asc_sort,
                )
            elif sort_by == "pathogenicity":
                filtered_mutations = sorted(
                    filtered_mutations,
                    key=lambda item: patho_rank.get(item["pathogenicity"] if isinstance(item, dict) and "pathogenicity" in item else "Low", 4),
                    reverse=not asc_sort,
                )
            elif sort_by == "evidence":
                filtered_mutations = sorted(
                    filtered_mutations,
                    key=lambda item: evidence_order.get(item["evidence_type"] if isinstance(item, dict) and "evidence_type" in item else "Literature evidence", 3),
                    reverse=not asc_sort,
                )
            else:
                filtered_mutations = sorted(
                    filtered_mutations,
                    key=lambda item: item["name"] if isinstance(item, dict) and "name" in item else "",
                    reverse=not asc_sort,
                )

            st.caption("Variant list shown as expandable cards to reduce repeated table/card content.")

            for variant_item in filtered_mutations:
                with st.expander(f"{variant_item['name']} ({variant_item['type']})"):
                    st.write(f"**Tier:** {variant_item.get('tier', 'Tier IV')}")
                    st.write(f"**Pathogenicity:** {variant_item['pathogenicity']}")
                    st.write(f"**Classification:** {variant_item.get('classification', 'VUS')}")
                    st.write(f"**Functional effect:** {variant_item['effect']}")
                    st.write(f"**Sensitivity signal:** {variant_item['significance']['sensitivity']}")
                    st.write(f"**Resistance signal:** {variant_item['significance']['resistance']}")
                    st.write(f"**Prognosis trend:** {variant_item['significance']['prognosis']}")
                    st.write(f"**Evidence type:** {variant_item.get('evidence_type', 'Literature evidence')}")
                    st.markdown(f"**PubMed references:** {pubmed_markdown(variant_item.get('pubmed_articles', ''))}")

        with pane_reco:
            st.markdown("<div class='section-bar section-rec'>Recommendations</div>", unsafe_allow_html=True)
            st.markdown("<div class='hint-line'>Hover chips and score pills for richer visual cues; open a regimen to inspect scientific basis.</div>", unsafe_allow_html=True)
            for idx, combo in enumerate(pareto_combos[:top_n], start=1):
                combo_name = " + ".join(combo["drug_names"])
                scientific = regimen_scientific_details(combo, variant, results["cancer_type"])
                if combo["weighted_score"] >= 2.0:
                    quality_tag = "<span class='rec-tag tag-optimal'>Strong composite fit</span>"
                elif combo["weighted_score"] >= 1.0:
                    quality_tag = "<span class='rec-tag tag-balanced'>Balanced tradeoff</span>"
                else:
                    quality_tag = "<span class='rec-tag tag-watch'>Watch safety/cost</span>"
                overlap_tag = "<span class='bubble bubble-good'>Pathway overlap present</span>" if scientific["pathway_overlap"] else "<span class='bubble bubble-warn'>Pathway overlap limited</span>"
                direct_tag = "<span class='bubble bubble-good'>Target-direct linkage</span>" if scientific["direct_target"] else "<span class='bubble bubble-warn'>Target linkage indirect</span>"
                resistance_tag = "<span class='bubble bubble-risk'>Resistance flag present</span>" if scientific["marker_hits"] else "<span class='bubble bubble-good'>No direct resistance hit</span>"
                st.markdown(f"<div class='rec-shell {'rec-shell-top' if idx == 1 else ''}'>", unsafe_allow_html=True)
                st.markdown(f"<div class='rec-title'><span class='rank-dot'>{idx}</span><h4 style='margin:0;'>{combo_name}</h4></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rec-tags'>{quality_tag}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class='score-strip'>
                        <div class='score-pill score-pill-main'><div class='score-label'>Composite score</div><div class='score-value'>{combo['weighted_score']:.2f}</div></div>
                        <div class='score-pill score-pill-risk'><div class='score-label'>Safety burden</div><div class='score-value'>{combo['toxicity']:.2f}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Why this regimen", expanded=(idx == 1)):
                    st.markdown(
                        f"""
                        <div class='rationale-card'>
                            <div class='bubble-row'>
                                {overlap_tag}
                                {direct_tag}
                                {resistance_tag}
                            </div>
                            <div class='rationale-grid'>
                                <div class='rationale-callout'><span class='rationale-k'>Selection basis</span><span class='rationale-v'>Pareto-optimal under current objective weights (efficacy {w_eff:.2f}, safety {w_tox:.2f}, cost {w_cost:.2f}) with composite score {combo['weighted_score']:.2f}.</span></div>
                                <div class='rationale-callout'><span class='rationale-k'>Mechanism</span><span class='rationale-v'>{scientific['mechanism_line']}</span></div>
                                <div class='rationale-callout'><span class='rationale-k'>Sensitivity</span><span class='rationale-v'>{scientific['sensitivity_line']}</span></div>
                                <div class='rationale-callout'><span class='rationale-k'>Resistance</span><span class='rationale-v'>{scientific['resistance_line']}</span></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Drug-level mechanistic details**")
                    for line in scientific["drug_lines"]:
                        st.markdown(f"- {line}")
                st.markdown("</div>", unsafe_allow_html=True)

        with pane_viz:
            st.markdown("<div class='section-bar section-viz'>Visual Analytics</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.2, 1.1])
            score_view = c1.selectbox("Score view", ["Weighted Score", "Raw Score"], index=0, key="viz_score_view")
            drug_mode = c2.selectbox("Drug chart mode", ["Frequency", "Share (%)"], index=0, key="viz_drug_mode")
            variant_focus = c3.selectbox("Variant chart focus", ["All visible variants", "Pathogenic variants only"], index=0, key="viz_variant_focus")
            max_window = min(10, max(1, top_n))
            window_default = min(5, max_window)
            top_window = c4.slider("Recommendation window", 1, max_window, window_default, key="viz_top_window")
            st.caption("Tip: click legend items to isolate series, drag to zoom, and double-click to reset view.")

            df = pd.DataFrame([
                {"Combination": " + ".join(combo["drug_names"]), "Weighted Score": combo["weighted_score"], "Raw Score": combo["raw_score"], "Toxicity": combo["toxicity"], "Cost": combo["cost"], "Status": "Optimal" if combo["is_optimal"] else "Dominated", "Type": combo["combo_type"]}
                for combo in all_combos
            ])

            z_axis = score_view
            fig3d = px.scatter_3d(
                df,
                x="Cost",
                y="Toxicity",
                z=z_axis,
                color="Status",
                symbol="Type",
                hover_name="Combination",
                hover_data={"Raw Score": ":.2f", "Weighted Score": ":.2f", "Toxicity": ":.2f", "Cost": ":.2f", "Type": True, "Status": True},
                color_discrete_map={"Optimal": "#23817f", "Dominated": "#8ea2b4"},
            )
            fig3d.update_traces(
                marker=dict(size=6, opacity=0.86, line=dict(width=0.4, color="#d7e3ec")),
                hovertemplate="<b>%{hovertext}</b><br>Cost burden: %{x:.2f}<br>Safety burden: %{y:.2f}<br>" + z_axis + ": %{z:.2f}<extra></extra>",
            )
            fig3d.update_layout(
                title=f"Regimen tradeoff landscape for {gene}<br><sup>{score_view} on Z-axis, lower cost/safety is favorable</sup>",
                scene=dict(
                    xaxis_title="Economic burden (lower is better)",
                    yaxis_title="Safety burden (lower is better)",
                    zaxis_title=f"{score_view} (higher is better)",
                    xaxis=dict(backgroundcolor="#f7fbfe", gridcolor="#dce7f0", zerolinecolor="#cedbe6"),
                    yaxis=dict(backgroundcolor="#f7fbfe", gridcolor="#dce7f0", zerolinecolor="#cedbe6"),
                    zaxis=dict(backgroundcolor="#f7fbfe", gridcolor="#dce7f0", zerolinecolor="#cedbe6"),
                ),
                legend_title_text="Pareto status",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                margin=dict(l=10, r=10, t=65, b=10),
            )
            st.plotly_chart(fig3d, use_container_width=True)

            if variant_focus == "Pathogenic variants only":
                focused_variants = [item for item in gene_variants if item.get("pathogenicity") in ["High", "Medium"]]
                if len(focused_variants) < 2:
                    focused_variants = [item for item in gene_variants if item.get("pathogenicity") == "High"]
            else:
                focused_variants = list(gene_variants)

            type_counts = pd.Series([item["type"] for item in focused_variants], name="Mutation Class").value_counts().reset_index() if focused_variants else pd.DataFrame(columns=["Mutation Class", "Count"])
            type_counts.columns = ["Mutation Class", "Count"]

            tier_order_viz = ["Tier I", "Tier II", "Tier III", "Tier IV"]
            patho_order_viz = ["High", "Medium", "Low"]
            tier_patho = pd.crosstab(
                pd.Series([item.get("tier", "Tier IV") for item in focused_variants], name="Tier"),
                pd.Series([item.get("pathogenicity", "Low") for item in focused_variants], name="Pathogenicity"),
            ).reindex(index=tier_order_viz, columns=patho_order_viz, fill_value=0)

            fig_variant = make_subplots(
                rows=1,
                cols=2,
                column_widths=[0.45, 0.55],
                horizontal_spacing=0.12,
                subplot_titles=("Variant Type Distribution", "Clinical Tier vs Pathogenicity"),
                specs=[[{"type": "bar"}, {"type": "heatmap"}]],
            )

            fig_variant.add_trace(
                go.Bar(
                    x=type_counts["Mutation Class"],
                    y=type_counts["Count"],
                    marker_color=["#2f7f7b", "#3f7298", "#bf8a3e", "#ae6171", "#6a8f56", "#7f6b9a"],
                    hovertemplate="Mutation class: %{x}<br>Count: %{y}<br>Share: %{customdata:.1%}<extra></extra>",
                    customdata=(type_counts["Count"] / max(1.0, sum(float(v) for v in type_counts["Count"].tolist()))) if not type_counts.empty else None,
                ),
                row=1,
                col=1,
            )
            fig_variant.add_trace(
                go.Heatmap(
                    z=tier_patho.values,
                    x=tier_patho.columns,
                    y=tier_patho.index,
                    colorscale=[[0.0, "#edf4f9"], [0.2, "#bad3e4"], [0.45, "#68a6b3"], [0.7, "#bf8a3e"], [1.0, "#924b63"]],
                    hovertemplate="Tier: %{y}<br>Pathogenicity: %{x}<br>Count: %{z}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            fig_variant.update_layout(
                title=f"Variant profiling for {gene}<br><sup>Distribution and clinical burden map</sup>",
                showlegend=False,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                margin=dict(l=20, r=20, t=65, b=20),
            )
            fig_variant.update_xaxes(title_text="Mutation Class", row=1, col=1)
            fig_variant.update_yaxes(title_text="Count", row=1, col=1)
            fig_variant.update_xaxes(title_text="Pathogenicity", row=1, col=2)
            fig_variant.update_yaxes(title_text="Clinical Tier", row=1, col=2, categoryorder="array", categoryarray=tier_order_viz[::-1])
            if not focused_variants:
                fig_variant.add_annotation(
                    text="No variants available for current focus",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(color="#5d6d7b", size=13),
                )
            st.plotly_chart(fig_variant, use_container_width=True)
            st.caption(f"Variant focus: {variant_focus} (n={len(focused_variants)} variants)")

            top_drug_counts = Counter()
            for combo in pareto_combos[:top_window]:
                top_drug_counts.update(combo["drug_names"])
            if top_drug_counts:
                drug_df = pd.DataFrame(
                    [{"Drug": drug, "Count": count} for drug, count in top_drug_counts.items()]
                ).sort_values(["Count", "Drug"], ascending=[False, True])
                total_count = max(1.0, sum(float(v) for v in drug_df["Count"].tolist()))
                drug_df["Share (%)"] = (drug_df["Count"] / total_count * 100).round(1)
                y_col = "Count" if drug_mode == "Frequency" else "Share (%)"
                fig_drug = px.bar(
                    drug_df,
                    x="Drug",
                    y=y_col,
                    title=f"Drug {drug_mode.lower()} across top {top_window} recommendations",
                    color="Count",
                    text="Count" if drug_mode == "Frequency" else "Share (%)",
                    color_continuous_scale=[[0.0, "#dbe8f2"], [0.45, "#63a2bf"], [1.0, "#2d7a6d"]],
                    hover_data={"Count": True, "Share (%)": True},
                )
                fig_drug.update_layout(
                    coloraxis_showscale=False,
                    hovermode="x unified",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    title=f"Drug {drug_mode.lower()} across top {top_window} recommendations<br><sup>Contribution concentration among shortlisted regimens</sup>",
                )
                fig_drug.update_traces(textposition="outside")
                fig_drug.update_yaxes(title_text=drug_mode)
                st.plotly_chart(fig_drug, use_container_width=True)
            else:
                st.info("No drug usage data available for the selected recommendation window.")


if __name__ == "__main__":
    main()
