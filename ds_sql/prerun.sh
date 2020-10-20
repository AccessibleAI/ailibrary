SUDO=''
if (( $EUID != 0 )); then
    SUDO='sudo'
fi
#postgresql libraries
$SUDO apt-get update
$SUDO apt-get install -y unixodbc-dev
#mysql libraries

$SUDO curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
$SUDO curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list


$SUDO apt-get update
$SUDO ACCEPT_EULA=Y apt-get install -y msodbcsql17

TEMP_DEB="$(mktemp)" &&
URL_DEB='https://downloads.mysql.com/archives/get/p/10/file/mysql-connector-odbc_8.0.21-1ubuntu18.04_amd64.deb'
wget -O "$TEMP_DEB" "$URL_DEB" &&
$SUDO dpkg -i "$TEMP_DEB"
rm -f "$TEMP_DEB"