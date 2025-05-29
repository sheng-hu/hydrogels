#! /bin/bash
for file in *.fa
do
    ../clustalo -i "$file" --outfmt=clu -o "../aln_files_fibrous/${file/.fa/.aln}" -v
done
