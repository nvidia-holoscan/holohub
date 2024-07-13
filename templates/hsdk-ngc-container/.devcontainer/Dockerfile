# syntax=docker/dockerfile:1

ARG HOLOSCAN_SDK_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v2.2.0-dgpu
FROM ${HOLOSCAN_SDK_IMAGE}

# Options for setup script
ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="true"
ARG USERNAME=holoscan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME /home/$USERNAME

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY common-debian.sh /tmp/common-debian.sh
RUN yes | unminimize 2>&1 \
    && bash /tmp/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/common-debian.sh

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# Install Holoscan CLI dependencies
RUN pip install $(sed -n '/done <<EOF/,/EOF/{/done <<EOF/b;/EOF/b;p}' /opt/nvidia/holoscan/bin/holoscan)

# Set the default user.
# (See https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)
USER $USERNAME
