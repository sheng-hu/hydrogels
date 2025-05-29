from Bio import AlignIO
from Bio.Align import AlignInfo


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
    "G": "1",
    "A": "1",
    "S": "2",
    "T": "2",
    "C": "2",
    "V": "3",
    "L": "3",
    "I": "3",
    "M": "3",
    "P": "3",
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
}


def pname(s):
    i, j = s
    return aa_property[i] + " - " + aa_property[j]
