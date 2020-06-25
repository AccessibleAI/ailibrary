SUDO=''
if (( $EUID != 0 )); then
    SUDO='sudo'
fi
$SUDO apt-get install unixodbc-dev
