#!/bin/bash

dirNames=$(find -maxdepth 1 -type d | awk -F"/" '{if (NF > 1) print $NF;}')
fileNumLimit=1500
parentDir=$(pwd)
DEBUG=0
GENERATE_OLD=0
function findIndex() {
    # OFS=$IFS
    # IFS="_"
    # names=$1
    # # index at the last
    # IFS=$OFS
    names=$1
    # strip all extension
    names=(${names//./ })
    names=${names[0]}
    names=(${names//_/ })
    len=${#names[*]}
    # echo "len: ${len}"
    echo ${names[${len}-1]}
}
# make for example CHOL_TRAJ20ps to CHOL_20ps
function getDirName() {
    names=$1
    names1=(${names//_/ })
    namePrefix=${names1[0]}
    names2=(${names//TRAJ/ })
    len=${#names2[*]}
    nameSuffix=${names2[${len}-1]}
    echo "${namePrefix}_${nameSuffix}"
}
for dir in ${dirNames[@]}
do
    if [ "${DEBUG}" -eq 1 ]; then
        echo "---------------"
        echo "new loop:, pwd"
        pwd
    fi
    currPath="${parentDir}/${dir}"
    cd "${dir}/TRAJ"
    subDirNames=$(find -maxdepth 1 -type d | awk -F"/" '{if (NF > 1) print $NF;}')  # Maybe write a separate function
    for subdir in ${subDirNames}
    do
        if [ "${DEBUG}" -eq 1 ]; then
            echo "before cd subdir, pwd:"
            pwd
        fi
        cd "${subdir}"
        count=0
        saveSubDirName=$(getDirName ${subdir})
        saveFilePath="${currPath}/MSD_DATA_CAMMP/MSD_${saveSubDirName}"
        if [ "${DEBUG}" -eq 1 ]; then
            echo "saveFilePath is ${saveFilePath}"
        fi
        if [ ! -d "${saveFilePath}" ]; then
            echo "${saveFilePath} does not exist, creat such path"
            if [ ! "${DEBUG}" -eq 1 ]; then
                mkdir -p "${saveFilePath}"
            fi                
        fi
        ## gromacs get the msd file
        fileNames=$(find -maxdepth 1 -name "*.xvg" -type f | awk -F"/" '{if (NF > 1) print $NF;}')
        for file in ${fileNames}
        do 
            if [ ${count} -lt "${fileNumLimit}" ]; then
                i=$(findIndex ${file})
                if [ "${DEBUG}" -eq 1 ]; then
                    # echo "will process ${file}"
                    dummy=1
                    if [ "${GENERATE_OLD}" -eq 0 ] && [ -f "${saveFilePath}/MSD_GMX_res_$i.xvg" ]; then
                        # echo "file already exists, will not process"
                        dummy=2
                    else
                        echo "will process ${file}"
                    fi
                else
                    # first check if file already exists, generate old file iff GENERATE_OLD is set and file already exists
                    if [ ! "${GENERATE_OLD}" -eq 0 ] || [ ! -f "${saveFilePath}/MSD_GMX_res_$i.xvg" ]; then
                        awk '{if($1!="#" && $1!="@" && NF==7){printf $1" %.12e %.12e %.12e\n", $2-$5,$3-$6,$4-$7}else{print $0}}' ${file} > ../traj_relative_res_$i.xvg
                        gmx analyze -f ../traj_relative_res_$i.xvg -msd "${saveFilePath}/MSD_GMX_res_$i.xvg" -time -nonormalize
                        rm ../traj_relative_res_$i.xvg
                    fi
                fi
                ((count++))
            else
                if [ "${DEBUG}" -eq 1 ]; then
                    echo "less files than fileNumLimit, pwd:"
                    pwd
                fi
                break
            fi
        done
        cd ..
    done
    cd "${parentDir}"
done
# testFileName="traj_nojump_80.xvg"
# indx=$(findIndex ${testFileName})
# echo $indx
# subdir="CHOL_TRAJ20ps"
# temp=$(getDirName ${subdir})
# echo $temp


## dunno it seems that somehow there are file like "/#MSD_GMX_res_4.xvg.1#"  file in the folder could clean it use
# find -name "*#" -type f -exec rm {} +

# Beware that ./generateMSD.sh >  awk '{if ($0 ~ !/^file*/) {print $0}}' wont work, need to convert that to use pipeline 
# so ./generateMSD.sh |  awk '{if ($0 ~ !/^file*/) {print $0}}'