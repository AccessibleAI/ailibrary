SUDO=''
if (( $EUID != 0 )); then
    SUDO='sudo'
fi
#postgresql awscli
$SUDO apt-get update
$SUDO apt-get install -y awscli
