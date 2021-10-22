POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -u|--url)
    URL="$2"
    shift # past argument
    shift # past value
    ;;
    -fn|--file_name)
    FILE_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;

esac
done

mkdir files
cd files
rm -R *
wget $URL -O $FILE_NAME

if [ $(cnvrg data link $DATASET | grep -q successfully) ]    ; then
    cnvrg data put $DATASET ./$FILE_NAME
else
    cnvrg data init --title=$DATASET
    cnvrg data put $DATASET ./$FILE_NAME
fi
