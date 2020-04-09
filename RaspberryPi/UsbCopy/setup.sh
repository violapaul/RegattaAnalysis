
function system_enable_service()
{
    if ! sudo systemctl is-enabled ${1}; then
        sudo systemctl enable ${1}
    fi
}

function system_disable_service()
{
    if sudo systemctl is-enabled ${1}; then
        sudo systemctl disable ${1}
    fi
}

function system_stop_service()
{
    if sudo systemctl is-active ${1}; then
        sudo systemctl stop ${1}
    fi
}

function system_start_service()
{
    sudo systemctl start ${1}
}


if ! diff usbcopy.service /etc/systemd/system/usbcopy.service; then
    echo "Updating usbcopy.service"
    sudo cp usbcopy.service /etc/systemd/system/usbcopy.service
    sudo chmod 664 /etc/systemd/system/usbcopy.service
    sudo systemctl daemon-reload
fi
