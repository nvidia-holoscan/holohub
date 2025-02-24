#!/bin/bash

/host/bin/rtienv-7.3.0 -l /opt/rti.com/rti_license.dat >> /tmp/rtienv.sh
chmod +x /tmp/rtienv.sh

source /tmp/rtienv.sh
exec /bin/bash