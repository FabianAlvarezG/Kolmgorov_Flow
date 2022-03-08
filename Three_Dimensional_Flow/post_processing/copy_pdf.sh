file_list=()
for i in {1..32}
do 
	file_list+=("N0128_rxy3_ryz4_"$i)
done

for file in "${file_list[@]}"
do
	scp lashgarak:/home/falvarez/ttf/temp/very_high_reynolds/ryz4/$file/figs/*.pdf ../post_processing/"$file"_stats.pdf
done
