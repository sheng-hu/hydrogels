#! /bin/bash
for file in *.fa
do
    ../clustalo -i "$file" --outfmt=clu -o "../aln_files/${file/.fa/.aln}" -v
done
