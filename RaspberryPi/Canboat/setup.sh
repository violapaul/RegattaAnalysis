
service="canboat.service"

function system_enable_service()
{
    if ! sudo systemctl is-enabled ${service}; then
        sudo systemctl enable ${service}
    fi
}

function system_disable_service()
{
    if sudo systemctl is-enabled ${service}; then
        sudo systemctl disable ${service}
    fi
}

function system_stop_service()
{
    if sudo systemctl is-active ${service}; then
        sudo systemctl stop ${service}
    fi
}

function system_start_service()
{
    sudo systemctl start ${service}
}

function reset()
{
    if ! diff canboat.service /etc/systemd/system/canboat.service; then
	sudo cp canboat.service /etc/systemd/system/canboat.service
	sudo chmod 664 /etc/systemd/system/canboat.service
	sudo systemctl daemon-reload
    fi
}
