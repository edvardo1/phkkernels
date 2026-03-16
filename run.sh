#!/bin/sh

M=1000
N=1000
K=1000

for i in "$@"; do
	if [ "$i" = "gemm" ];      then OPS="$OPS gemm"
	elif [ "$i" = "gevm" ];    then OPS="$OPS gevm"
	elif [ "$i" = "gemv" ];    then OPS="$OPS gemv"
	elif [ "$i" = "phk" ];     then RUN="$RUN phk"
	elif [ "$i" = "cuda" ];    then RUN="$RUN cuda"
	elif [ "$i" = "torch" ];   then RUN="$RUN torch"
	elif [ "$i" = "pr" ]; then
		PRINT_RESULT="true"
	elif echo "$i" | grep '^m='; then
		M=${i#m=}
	elif echo "$i" | grep '^n='; then
		N=${i#n=}
	elif echo "$i" | grep '^k='; then
		K=${i#k=}
	fi
done

if [ -z "$OPS" ]; then
	echo 'no operation chosen'
	exit 1
fi

if [ -z "$RUN" ]; then
	echo 'no run chosen'
	exit 1
fi
for r in $RUN; do
	for op in $OPS; do
		case $r in
			phk)
				#phr <(cat "phk_$op.ex" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g;s/finput_print_result/false/g")
				if [ "$PRINT_RESULT" = true ]; then pr="true";
				else                             pr="false"; fi
				cat "phk_$op.ex" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g;s/finput_print_result/$pr/g" > tmp.ex
				phr tmp.ex
				rm tmp.ex
				#echo running PHK
				#cat "phk_$op.ex" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g"
				;;
			cuda)
				if [ "$PRINT_RESULT" = true ]; then pr="true";
				else                               pr="false"; fi
				cat "cuda_$op.cu" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g;s/finput_print_result/$pr/g" > tmp.cu
				nvcrun tmp.cu
				rm tmp.cu
				#echo running CUDA
				#cat "cuda_$op.cu" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g"
				;;
			torch)
				if [ "$PRINT_RESULT" = true ]; then pr="True";
				else                              pr="False"; fi
				cat "torch_$op.py" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g;s/finput_print_result/$pr/g" > tmp.py
				python tmp.py
				rm tmp.py
				#echo running TORCH
				#cat "torch_$op.py" | sed "s/finput_m/$M/g;s/finput_n/$N/g;s/finput_k/$K/g"
				;;
		esac
		echo
	done
done

