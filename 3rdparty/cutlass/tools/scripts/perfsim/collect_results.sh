echo -e "ID \t SM_SOL% \t NVPDM"
for line in $(find . | grep perfsim/Chip.json.gz); do
  clks=`zgrep -B 1 -i elapsedclocks\" $line | head -n 1 | cut -d ":" -f 2 | cut -d "," -f 1`
  sol=`zgrep -A 2 -i '\"name\" \: \"SM\"' $line | tail -n 1 | cut -d ":" -f 2 | cut -d '"' -f 2`
  id=`echo $line | cut -d "/" -f5`
  test_id=`echo $id | cut -d "." -f 1`
  name=`cat ./perfsim/apic_capture/run.A.dir.0/$test_id/state.yml | grep workload | cut -d ":" -f 2`
  link=`echo "=HYPERLINK(\"https://us-edaws.nvidia.com/browse/home/sc-edaws/nvpdm/index.html#/perfsim?rp=$PWD/$line\",\"link\")" | sed 's/perfsim\/Chip\.json\.gz//g'`
  echo -e "$name \t $sol \t $link"
  #echo $line
done
