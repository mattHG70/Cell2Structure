#!/bin/bash

# Slurm job to download all image archives
# unzip the archives
# remove the zip file after extraction

# Slurm PARAMETERS
#SBATCH --job-name=capst_team6_gi
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1g
#SBATCH --account=siads699f24_class
#SBATCH --mail-user mhuebsch@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/get_images_%x-%j.log
#SBATCH --error=/home/%u/get_images_error-%x-%j.log

# run the image retrieval and uppacking as a cluster job
# due to its run time. No blocking of resources on the login node

# run bashrc for the current user
source /home/mhuebsch/.bashrc

# actual image directory on the SIADS 699 scratch space
# adapt the user folder when needed
mkdir /scratch/siads699f24_class_root/siads699f24_class/mhuebsch/images
cd /scratch/siads699f24_class_root/siads699f24_class/mhuebsch/images

START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow

wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip
unzip -q BBBC021_v1_images_Week1_22123.zip
rm BBBC021_v1_images_Week1_22123.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22141.zip
unzip -q BBBC021_v1_images_Week1_22141.zip
rm BBBC021_v1_images_Week1_22141.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22161.zip
unzip -q BBBC021_v1_images_Week1_22161.zip
rm BBBC021_v1_images_Week1_22161.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22361.zip
unzip -q BBBC021_v1_images_Week1_22361.zip
rm BBBC021_v1_images_Week1_22361.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22381.zip
unzip -q BBBC021_v1_images_Week1_22381.zip
rm BBBC021_v1_images_Week1_22381.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22401.zip
unzip -q BBBC021_v1_images_Week1_22401.zip
rm BBBC021_v1_images_Week1_22401.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24121.zip
unzip -q BBBC021_v1_images_Week2_24121.zip
rm BBBC021_v1_images_Week2_24121.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24141.zip
unzip -q BBBC021_v1_images_Week2_24141.zip
rm BBBC021_v1_images_Week2_24141.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24161.zip
unzip -q BBBC021_v1_images_Week2_24161.zip
rm BBBC021_v1_images_Week2_24161.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24361.zip
unzip -q BBBC021_v1_images_Week2_24361.zip
rm BBBC021_v1_images_Week2_24361.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24381.zip
unzip -q BBBC021_v1_images_Week2_24381.zip
rm BBBC021_v1_images_Week2_24381.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week2_24401.zip
unzip -q BBBC021_v1_images_Week2_24401.zip
rm BBBC021_v1_images_Week2_24401.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25421.zip
unzip -q BBBC021_v1_images_Week3_25421.zip
rm BBBC021_v1_images_Week3_25421.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25441.zip
unzip -q BBBC021_v1_images_Week3_25441.zip
rm BBBC021_v1_images_Week3_25441.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25461.zip
unzip -q BBBC021_v1_images_Week3_25461.zip
rm BBBC021_v1_images_Week3_25461.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25681.zip
unzip -q BBBC021_v1_images_Week3_25681.zip
rm BBBC021_v1_images_Week3_25681.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25701.zip
unzip -q BBBC021_v1_images_Week3_25701.zip
rm BBBC021_v1_images_Week3_25701.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week3_25721.zip
unzip -q BBBC021_v1_images_Week3_25721.zip
rm BBBC021_v1_images_Week3_25721.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week4_27481.zip
unzip -q BBBC021_v1_images_Week4_27481.zip
rm BBBC021_v1_images_Week4_27481.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week4_27521.zip
unzip -q BBBC021_v1_images_Week4_27521.zip
rm BBBC021_v1_images_Week4_27521.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week4_27542.zip
unzip -q BBBC021_v1_images_Week4_27542.zip
rm BBBC021_v1_images_Week4_27542.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week4_27801.zip
unzip -q BBBC021_v1_images_Week4_27801.zip
rm BBBC021_v1_images_Week4_27801.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week4_27821.zip
unzip -q BBBC021_v1_images_Week4_27821.zip
rm BBBC021_v1_images_Week4_27821.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week4_27861.zip
unzip -q BBBC021_v1_images_Week4_27861.zip
rm BBBC021_v1_images_Week4_27861.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week5_28901.zip
unzip -q BBBC021_v1_images_Week5_28901.zip
rm BBBC021_v1_images_Week5_28901.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week5_28921.zip
unzip -q BBBC021_v1_images_Week5_28921.zip
rm BBBC021_v1_images_Week5_28921.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week5_28961.zip
unzip -q BBBC021_v1_images_Week5_28961.zip
rm BBBC021_v1_images_Week5_28961.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week5_29301.zip
unzip -q BBBC021_v1_images_Week5_29301.zip
rm BBBC021_v1_images_Week5_29301.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week5_29321.zip
unzip -q BBBC021_v1_images_Week5_29321.zip
rm BBBC021_v1_images_Week5_29321.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week5_29341.zip
unzip -q BBBC021_v1_images_Week5_29341.zip
rm BBBC021_v1_images_Week5_29341.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week6_31641.zip
unzip -q BBBC021_v1_images_Week6_31641.zip
rm BBBC021_v1_images_Week6_31641.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week6_31661.zip
unzip -q BBBC021_v1_images_Week6_31661.zip
rm BBBC021_v1_images_Week6_31661.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week6_31681.zip
unzip -q BBBC021_v1_images_Week6_31681.zip
rm BBBC021_v1_images_Week6_31681.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week6_32061.zip
unzip -q BBBC021_v1_images_Week6_32061.zip
rm BBBC021_v1_images_Week6_32061.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week6_32121.zip
unzip -q BBBC021_v1_images_Week6_32121.zip
rm BBBC021_v1_images_Week6_32121.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week6_32161.zip
unzip -q BBBC021_v1_images_Week6_32161.zip
rm BBBC021_v1_images_Week6_32161.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week7_34341.zip
unzip -q BBBC021_v1_images_Week7_34341.zip
rm BBBC021_v1_images_Week7_34341.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week7_34381.zip
unzip -q BBBC021_v1_images_Week7_34381.zip
rm BBBC021_v1_images_Week7_34381.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week7_34641.zip
unzip -q BBBC021_v1_images_Week7_34641.zip
rm BBBC021_v1_images_Week7_34641.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week7_34661.zip
unzip -q BBBC021_v1_images_Week7_34661.zip
rm BBBC021_v1_images_Week7_34661.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week7_34681.zip
unzip -q BBBC021_v1_images_Week7_34681.zip
rm BBBC021_v1_images_Week7_34681.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week8_38203.zip
unzip -q BBBC021_v1_images_Week8_38203.zip
rm BBBC021_v1_images_Week8_38203.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week8_38221.zip
unzip -q BBBC021_v1_images_Week8_38221.zip
rm BBBC021_v1_images_Week8_38221.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week8_38241.zip
unzip -q BBBC021_v1_images_Week8_38241.zip
rm BBBC021_v1_images_Week8_38241.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week8_38341.zip
unzip -q BBBC021_v1_images_Week8_38341.zip
rm BBBC021_v1_images_Week8_38341.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week8_38342.zip
unzip -q BBBC021_v1_images_Week8_38342.zip
rm BBBC021_v1_images_Week8_38342.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week9_39206.zip
unzip -q BBBC021_v1_images_Week9_39206.zip
rm BBBC021_v1_images_Week9_39206.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week9_39221.zip
unzip -q BBBC021_v1_images_Week9_39221.zip
rm BBBC021_v1_images_Week9_39221.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week9_39222.zip
unzip -q BBBC021_v1_images_Week9_39222.zip
rm BBBC021_v1_images_Week9_39222.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week9_39282.zip
unzip -q BBBC021_v1_images_Week9_39282.zip
rm BBBC021_v1_images_Week9_39282.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week9_39283.zip
unzip -q BBBC021_v1_images_Week9_39283.zip
rm BBBC021_v1_images_Week9_39283.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week9_39301.zip
unzip -q BBBC021_v1_images_Week9_39301.zip
rm BBBC021_v1_images_Week9_39301.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week10_40111.zip
unzip -q BBBC021_v1_images_Week10_40111.zip
rm BBBC021_v1_images_Week10_40111.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week10_40115.zip
unzip -q BBBC021_v1_images_Week10_40115.zip
rm BBBC021_v1_images_Week10_40115.zip
wget -nv https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week10_40119.zip
unzip -q BBBC021_v1_images_Week10_40119.zip
rm BBBC021_v1_images_Week10_40119.zip

EXITCODE=$?

END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
