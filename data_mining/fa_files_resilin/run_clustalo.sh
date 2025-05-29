#! /bin/bash
for file in *.fa
do
    ../clustalo -i "$file" --outfmt=clu -o "../aln_files_resilin/${file/.fa/.aln}" -v
done
