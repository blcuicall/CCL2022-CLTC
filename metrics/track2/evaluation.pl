use utf8;
use Encode;
#to seperate the submission into traditional eva aspects(d i p) and correction aspect 2018.4.25
#基于2018年的my_evaluation.pl修改而来。
#修正identification层面的的错误。Identification的原意是在一个单元里，有几类错误（而不是几个），看能不能找全。如果一个单元里SSMR，是算3而不是4。实际评测计算时变成了基于错误点的。也就是SSMR是按照4，来计算pre和rec了。所以其实是压低了成绩。
#position层面的计算是终极的，位置、类型全对才对。
#2021年的改进是去除了top1的track只保留top3，而答案也有1-3个，所以在erro_cor里也做了修改
#要思考一个unit矛盾答复的情况：12346,correct  12346,3,4,M 又挑错，又说句子对。怎么办。其实有很多神经系统会如此抽风。

#参数0是待评测结果，参数1是输出的报告文件，参数2是答案##

open(out,">$ARGV[1]");#eva-report
open(in,,'<:encoding(utf8)',"$ARGV[2]");#truth	
while(<in>){
	chomp;
	if(/^(\S+)\,\s+correct/){
		$corr{$1}=1;
	}elsif(/^(\S+)\,\s+(\d+)\,\s+(\d+)\,\s+([RWMS])/){
		$erro_exsist{$1}=1;
		$erro_ident{$1.' '.$4}=1;
		$erro_pos{$1.' '.$2.' '.$3.' '.$4}=1;
	}else{
		#print $_;
	}
	$truth_count++;
	
	if(/^(\S+)\,\s+(\d+)\,\s+(\d+)\,\s+([MS])\,\s+(\S.*)/){#为修正track预备
		$id=$1; $st=$2; $en=$3; $ty=$4; $anses=$5;
		if($anses=~/\,/){
			@tmp=$anses=~/([^\,\s]+)/g;
			foreach $k (@tmp){
				$erro_cor{$id.' '.$st.' '.$en.' '.$ty.' '.$k}=1;
			}
		}else{
			$erro_cor{$id.' '.$st.' '.$en.' '.$ty.' '.$anses}=1;
		}
	}
}
close(in);

$corr_count=keys %corr;
$erro_exsist_count=keys %erro_exsist;
$erro_ident_count=keys %erro_ident;
$erro_pos_count=keys %erro_pos;
$erro_cor_count=keys %erro_cor;
print "Correct Units: $corr_count\tUnits With Errors: $erro_exsist_count\nError Type Counts in All Units: $erro_ident_count\tError Count: $erro_pos_count\tErrors Need Correction: $erro_cor_count";
print "\n\n=====================\n\n";
#输出一个表头
print out "CGED2021 Evaluation\n\nSystem Output:$ARGV[0]\nGold Standard: $ARGV[2]\n\n=====================\n\n";
print out "Correct Units: $corr_count\tUnits With Errors: $erro_exsist_count\nError Type Counts in All Units: $erro_ident_count\tError Count: $erro_pos_count\tError-Corrections: $erro_cor_count";
print out "\n\n=====================\n\n";

open(in,'<:encoding(utf8)',"$ARGV[0]");#system output
while($line=<in>){
	chomp $line;
	if($line=~/^(\S+)\,\s+correct/){
		unless(defined $hash{$1}){ #预防即标错误，又标correct的矛盾标注
			$hash{$1}='correct';
			$hash_corr{$1}=1;
		}
	}elsif($line=~/^(\S+)\,\s+(\d+)\,\s+(\d+)\,\s+([RWMS])/){		
		$hash{$1}=1;
		$hash_exsist{$1}=1;
		$hash_ident{$1.' '.$4}++;
		$hash_pos{$1.' '.$2.' '.$3.' '.$4}++;
		if(defined $hash_corr{$1}){ #即标错误，又标correct的矛盾标注都按照其标注的错误来处理，不算correct
			delete $hash_corr{$1};
		}
	}else{
		#print encode('gbk',$base_info.' '.$e."\n");
	}
	
	if($line=~/^(\S+)\,\s+(\d+)\,\s+(\d+)\,\s+([MS])\,\s+(\S.*)/){#为修正track预备
		$base_info=$1.' '.$2.' '.$3.' '.$4;
		$ans_s=$5;
		@ans=$ans_s=~/([^\,\s]+)/g;
		foreach $e (@ans){
			$hash_cor_top3{$base_info.' '.$e}=1;
			#print encode('gbk',$base_info.' '.$e."\n");
		}
	}
}
close(in);

$hash_corr_count=keys %hash_corr;
$hash_exsist_count=keys %hash_exsist;
$hash_ident_count=keys %hash_ident;
$hash_pos_count=keys %hash_pos;
$hash_cor_top3_count=keys %hash_cor_top3;
print "System Correct Units: $hash_corr_count\tSystem Units With Errors: $hash_exsist_count\nSystem Error Type Counts in All Units: $hash_ident_count\tSystem Error Count: $hash_pos_count\tSystem Corrected Erros:$hash_cor_top3_count";
print "\n\n=====================\n\n";
print out "System Correct Units: $hash_corr_count\tSystem Units With Errors: $hash_exsist_count\nSystem Error Type Counts in All Units: $hash_ident_count\tSystem Error Count: $hash_pos_count\tSystem Corrected Erros:$hash_cor_top1_count";
print out "\n\n=====================\n\n";

#计算假阳性FPR
foreach $e (sort keys %corr){
	if(defined $hash{$e}){
		if($hash{$e} ne 'correct'){
			$fp_exsist++;
		}else{
			$tn_exsist++;
		}
	}else{#如果系统没有对某句话做任何判断，也不能算它对
		$fp_exsist++;
	}
}
$fpr=$fp_exsist/$corr_count;
print "False Positive Rate = $fpr ( $fp_exsist / $corr_count)";
print "\n\n=====================\n\n";
$fpr=sprintf('%.4f',$fpr);
print out "False Positive Rate = $fpr ( $fp_exsist / $corr_count)";
print out "\n\n=====================\n\n";


#计算detection层面
foreach $e (sort keys %erro_exsist){
	if(defined $hash{$e}){
		if($hash{$e} ne 'correct'){
			$tp_exsist++;
		}else{
			$fn_exsist++;
		}
	}else{ #没有作答的句子就算错误
		#print $e."!!!\n";
		$fn_exsist++;
	}
}
$pre_exsist=$tp_exsist/$hash_exsist_count;
$rec_exsist=$tp_exsist/$erro_exsist_count;
$f1_detection=2*$pre_exsist*$rec_exsist/($pre_exsist+$rec_exsist);

format_opt($pre_exsist,$tp_exsist,$hash_exsist_count,$rec_exsist,$erro_exsist_count,$f1_detection,'Detection Level');
print "Detction Level\nPre = $pre_exsist ($tp_exsist / $hash_exsist_count)\nRec = $rec_exsist ($tp_exsist / $erro_exsist_count)\nF1 = $f1_detection (2* $pre_exsist * $rec_exsist /( $pre_exsist + $rec_exsist ))";	
print "\n\n=====================\n\n";

#计算identification层面
open(out2,">$ARGV[1].ident");
foreach $e (sort keys %erro_ident){
	if(defined $hash_ident{$e}){
		$tp_ident++;
		print out2 $tp_ident." $e\n";
	}else{ #没有作答的句子就算错误
		$fn_ident++;
	}
}
close(out2);
$pre_ident=$tp_ident/$hash_ident_count;
$rec_ident=$tp_ident/$erro_ident_count;
$f1_identification=2*$pre_ident*$rec_ident/($pre_ident+$rec_ident);
format_opt($pre_ident,$tp_ident,$hash_ident_count,$rec_ident,$erro_ident_count,$f1_identification,'Identification Level');
print "Identification Level\nPre = $pre_ident ($tp_ident / $hash_ident_count)\nRec = $rec_ident ($tp_ident / $erro_ident_count)\nF1 = $f1_identification (2* $pre_ident * $rec_ident / ( $pre_ident+$rec_ident ))";	
print "\n\n=====================\n\n";

#计算position层面
foreach $e (sort keys %erro_pos){
	if(defined $hash_pos{$e}){
		$tp_pos++;
	}else{ #没有作答的句子就算错误
		$fn_pos++;
	}
}
$pre_pos=$tp_pos/$hash_pos_count;
$rec_pos=$tp_pos/$erro_pos_count;
if($pre_pos+$rec_pos>0){
	$f1_position=2*$pre_pos*$rec_pos/($pre_pos+$rec_pos);
}else{
	$f1_position=0;
}
format_opt($pre_pos,$tp_pos,$hash_pos_count,$rec_pos,$erro_pos_count,$f1_position,'Position Level');
print "Postion Level\nPre = $pre_pos ( $tp_pos / $hash_pos_count )\nRec = $rec_pos ( $tp_pos / $erro_pos_count )\nF1 = $f1_position ( 2 * $pre_pos * $rec_pos /( $pre_pos+$rec_pos ) )";
print "\n\n=====================\n\n";

#计算修正track
foreach $e (sort keys %erro_cor){
	#print encode('gbk',$e."\n");
	if(defined $hash_cor_top3{$e}){
		$hit_top3++;
	}
}

if($hash_cor_top3_count>0){
	$pre_top3=$hit_top3/$hash_cor_top3_count;
	$rec_top3=$hit_top3/$erro_cor_count;	
}
if($pr_top3+$rec_top3>0){
	$f1_top3=2*$pre_top3*$rec_top3/($pre_top3+$rec_top3);
}else{
	$f1_top3=0;
}

format_opt($pre_top3,$hit_top3,$hash_cor_top3_count,$rec_top3,$erro_cor_count,$f1_top3,'Correction Level');
print "Correction Level\nPre = $pre_top3 ( $hit_top3 / $hash_cor_top3_count )\nRec = $rec_top3 ( $hit_top3 / $erro_cor_count )\nF1 = $f1_top3 ( 2 * $pre_top3 * $rec_top3 /( $pre_top3 + $rec_top3 ) )\n";

$comprehensive_score=0.25*$f1_detection+0.25*$f1_identification+0.25*$f1_position+0.25*$f1_top3-0.25*$fpr;
print "\nComprehensive score = $comprehensive_score";
print "\n\n==========END===========\n\n";


#格式输出，保留小数点后四位
#format_opt($pre_exsist,$tp_exsist,$hash_exsist_count,$rec_exsist,$erro_exsist_count,$f1_detection)
sub format_opt{
	my $pre1=@_[0];
	my $tp1=@_[1];
	my $hash_count1=@_[2];
	my $rec1=@_[3];
	my $erro_count1=@_[4];
	my $f11=@_[5];
	my $name=@_[6];
	
	$pre=sprintf('%.4f',$pre1);
	$tp=$tp1;
	$hash_count=$hash_count1;
	$rec=sprintf('%.4f',$rec1);
	$erro_count=$erro_count1;
	$f1=sprintf('%.4f',$f11);
	
	print out "$name\nPre = $pre ($tp / $hash_count)\nRec = $rec ($tp / $erro_count)\nF1 = $f1 (2* $pre * $rec /( $pre + $rec ))";	
	print out "\n\n=====================\n\n";
	
}