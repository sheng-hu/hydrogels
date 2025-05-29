import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from itertools import groupby
import seaborn as sns
from Bio import SeqIO


class obj:
    def __init__(self, rec):
        self.id = rec.id
        self.name = rec.name
        self.features = rec.features  # list[SeqFeature]
        self.description = rec.description
        self.taxonomy = rec.annotations["taxonomy"]
        self.organism = rec.annotations["organism"]
        self.source = rec.annotations["source"]
        self.seq = rec.seq  # Seq

    def __repr__(self):
        return f"{self.name} ({len(self.seq)} aa, {len(self.features)} feat) {self.description}"


aa = "ARNDCQEGHILKMFPSTWYV"

aa_property = {
    "1": "small",
    "2": "nucleophilic",
    "3": "hydrophobic",
    "4": "aromatic",
    "5": "acidic",
    "6": "amide",
    "7": "basic",
}

aasub = {
    "S": "2",
    "T": "2",
    "C": "2",
    "V": "3",
    "L": "3",
    "I": "3",
    "M": "3",
    "P": "3",
    "G": "3",
    "A": "3",
    "F": "4",
    "Y": "4",
    "W": "4",
    "D": "5",
    "E": "5",
    "N": "6",
    "Q": "6",
    "H": "7",
    "K": "7",
    "R": "7",
    "B": "",
    "J": "",
    "Z": "",
}
# aasub = {'G': '1', 'A': '1',
#          'S': '2', 'T': '2', 'C': '2',
#          'V': '3', 'L': '3', 'I': '3', 'M': '3', 'P': '3',
#          'F': '4', 'Y': '4', 'W': '4',
#          'D': '5', 'E': '5',
#          'N': '6', 'Q': '6',
#          'H': '7', 'K': '7', 'R': '7',
#          'B': '', 'J': '', 'Z': '',
#          }


def pname(s):
    i, j = s
    return aa_property[i] + " - " + aa_property[j]


def singlename(s):
    i = s
    return aa_property[i]


def check_aa_ratios_overal(l, aa):
    count_sum = 0
    overal_sum = 0
    for seq in l:
        count_sum += str(seq).count(aa)
        overal_sum += len(str(seq))
    return count_sum, overal_sum, count_sum / overal_sum


def get_consensus_len(key, remove_x=False):

    consensus = SeqIO.read(f"con_files/{key}.txt", "fasta")
    # with open(f'con_files/{k}.txt', 'w') as f:
    #     print(SeqRecord(consensus, header, '', '').format('fasta'), file=f)
    # producing pair_files/*
    tseq = str(consensus.seq)
    n_tseq = tseq

    if remove_x:
        n_tseq = [cc for cc in tseq if cc not in "XAG"]  # in 'X'
    return len(n_tseq)


def process_str(tseq):
    n_tseq = [cc for cc in tseq if cc not in "XAG"]  # in 'X'
    # Group the A into 6 categories.
    pseq = "".join([aasub[b] for b in n_tseq])
    # 2-gram sliding along the index 0.
    single = [singlename(pseq[i]) for i in range(len(pseq))]
    c = collections.Counter(single)
    # c.most_common()
    # with open(f'single_files/{k}.txt', 'w') as f:
    #     print(c.most_common(), file=f)
    sum_counts = 0
    for x, y in c.most_common():
        sum_counts += y

    aa_dict = dict.fromkeys(
        ["hydrophobic", "nucleophilic", "aromatic", "acidic", "amide", "basic"], 0
    )

    for x, y in c.most_common():
        aa_dict[x] = y

    return aa_dict, sum_counts


def process_str_include_small(tseq):
    n_tseq = [cc for cc in tseq if cc not in "X"]  # in 'X'
    # Group the A into 6 categories.
    pseq = "".join([aasub[b] for b in n_tseq])
    # 2-gram sliding along the index 0.
    single = [singlename(pseq[i]) for i in range(len(pseq))]
    c = collections.Counter(single)
    # c.most_common()
    # with open(f'single_files/{k}.txt', 'w') as f:
    #     print(c.most_common(), file=f)
    sum_counts = 0
    for x, y in c.most_common():
        sum_counts += y

    aa_dict = dict.fromkeys(
        [
            "hydrophobic",
            "nucleophilic",
            "aromatic",
            "acidic",
            "amide",
            "basic",
            "small",
        ],
        0,
    )

    for x, y in c.most_common():
        aa_dict[x] = y

    return aa_dict, sum_counts


def process_pair_str(tseq):

    n_tseq = [cc for cc in tseq if cc not in "XAG"]  # in 'X'
    # Group the A into 6 categories.
    pseq = "".join([aasub[b] for b in n_tseq])
    # 2-gram sliding along the index 0.
    # single = [pname(pseq[i]) for i in range(len(pseq))]
    pair = [pname(sorted(pseq[i : i + 2])) for i in range(len(pseq) - 1)]
    c = collections.Counter(pair)
    pair_dict = dict((x, y) for x, y in c.most_common())
    # print(pair_dict)

    return pair_dict


def process_pair_str_include_small(tseq):

    n_tseq = [cc for cc in tseq if cc not in "X"]  # in 'X'
    # Group the A into 6 categories.
    pseq = "".join([aasub[b] for b in n_tseq])
    # 2-gram sliding along the index 0.
    # single = [pname(pseq[i]) for i in range(len(pseq))]
    pair = [pname(sorted(pseq[i : i + 2])) for i in range(len(pseq) - 1)]
    c = collections.Counter(pair)
    pair_dict = dict((x, y) for x, y in c.most_common())
    # print(pair_dict)

    return pair_dict


def process_pair_list(seq_list):
    total_dict = defaultdict(int)
    for seq in seq_list:
        pair_dict = process_pair_str(seq.seq)
        for k in pair_dict.keys():
            if not total_dict.get(k, None):
                total_dict[k] = 0
            total_dict[k] += pair_dict[k]
    return total_dict


def process_pair_list_include_small(seq_list):
    total_dict = defaultdict(int)
    for seq in seq_list:
        pair_dict = process_pair_str_include_small(seq.seq)
        for k in pair_dict.keys():
            if not total_dict.get(k, None):
                total_dict[k] = 0
            total_dict[k] += pair_dict[k]
    return total_dict


def process_seq_list(seq_list):
    sum_counts = 0
    aa_dict = dict.fromkeys(
        ["hydrophobic", "nucleophilic", "aromatic", "acidic", "amide", "basic"], 0
    )
    for seq in seq_list:
        seq_dict, seq_counts = process_str(seq.seq)
        sum_counts += seq_counts
        # print(seq_dict['hydrophobic'])
        # print(aa_dict['hydrophobic'])
        aa_dict["hydrophobic"] += seq_dict["hydrophobic"]
        aa_dict["nucleophilic"] += seq_dict["nucleophilic"]
        aa_dict["aromatic"] += seq_dict["aromatic"]
        aa_dict["acidic"] += seq_dict["acidic"]
        aa_dict["amide"] += seq_dict["amide"]
        aa_dict["basic"] += seq_dict["basic"]

    return aa_dict, sum_counts


def process_seq_list_not_classified(seq_list):
    sum_counts_external = 0
    default_keys = aasub.keys()
    aa_dict_external = dict.fromkeys(default_keys, 0)
    for seq in seq_list:

        n_tseq = [cc for cc in seq.seq if cc not in "X"]  # in 'X'
        c = collections.Counter(n_tseq)
        # c.most_common()
        # with open(f'single_files/{k}.txt', 'w') as f:
        #     print(c.most_common(), file=f)
        sum_counts = 0
        for x, y in c.most_common():
            sum_counts += y

        aa_dict = dict.fromkeys(aasub.keys(), 0)

        for x, y in c.most_common():
            aa_dict[x] = y

        sum_counts_external += sum_counts
        for k in aasub.keys():
            aa_dict_external[k] += aa_dict[k]

    return aa_dict_external, sum_counts_external


def process_seq_list_include_small(seq_list):
    sum_counts = 0
    aa_dict = dict.fromkeys(
        [
            "hydrophobic",
            "nucleophilic",
            "aromatic",
            "acidic",
            "amide",
            "basic",
            "small",
        ],
        0,
    )
    for seq in seq_list:
        seq_dict, seq_counts = process_str_include_small(seq.seq)
        sum_counts += seq_counts
        # print(seq_dict['hydrophobic'])
        # print(aa_dict['hydrophobic'])
        aa_dict["hydrophobic"] += seq_dict["hydrophobic"]
        aa_dict["nucleophilic"] += seq_dict["nucleophilic"]
        aa_dict["aromatic"] += seq_dict["aromatic"]
        aa_dict["acidic"] += seq_dict["acidic"]
        aa_dict["amide"] += seq_dict["amide"]
        aa_dict["basic"] += seq_dict["basic"]
        aa_dict["small"] += seq_dict["small"]

    return aa_dict, sum_counts


def fill_heatmap_mat(tseq):
    # tseq = str(consensus.seq)

    # Note here is the counting rules for common sequence.
    # Remove those in 'XAGP'
    n_tseq = [cc for cc in tseq if cc not in "XAG"]  # in 'X'
    # Group the A into 6 categories.
    pseq = "".join([aasub[b] for b in n_tseq])
    groups = groupby(pseq)
    partitioned_list = []
    for key, g in groups:
        partitioned_list.append("".join(list(g)))
    partitioned_c = collections.Counter(partitioned_list)
    mc = partitioned_c.most_common()
    block_max = max([len(x) for x, _ in mc])
    block_max = 10

    # Note that df.plot(kind=) in default use the first column as the index, i.e. the x axis.
    # This time not use defaultdict(list), but use np.zeros to generte the ndarrays to store data and fill it out.
    mat = np.zeros((block_max, 7))
    mat[:, 0] = np.array(list(range(1, block_max + 1)))
    for x, y in mc:
        if len(x) - 1 < block_max:
            mat[len(x) - 1, int(x[0]) - 1] = y  # x[0]-1 as we remove small 1

    return mat


def fill_heatmap_mat_include_small(tseq):
    # tseq = str(consensus.seq)

    # Note here is the counting rules for common sequence.
    # Remove those in 'XAGP'
    n_tseq = [cc for cc in tseq if cc not in "X"]  # in 'X'
    # Group the A into 6 categories.
    pseq = "".join([aasub[b] for b in n_tseq])
    groups = groupby(pseq)
    partitioned_list = []
    for key, g in groups:
        partitioned_list.append("".join(list(g)))
    partitioned_c = collections.Counter(partitioned_list)
    mc = partitioned_c.most_common()
    block_max = max([len(x) for x, _ in mc])
    block_max = 10

    # Note that df.plot(kind=) in default use the first column as the index, i.e. the x axis.
    # This time not use defaultdict(list), but use np.zeros to generte the ndarrays to store data and fill it out.
    mat = np.zeros((block_max, 8))
    mat[:, 0] = np.array(list(range(1, block_max + 1)))
    for x, y in mc:
        if len(x) - 1 < block_max:
            mat[len(x) - 1, int(x[0])] = y  # x[0]-1 as we remove small 1
    return mat
