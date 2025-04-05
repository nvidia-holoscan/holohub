FROM squidfunk/mkdocs-material:latest

# Install plugins.
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh
RUN apk add gcc python3-dev musl-dev linux-headers
COPY doc/website/requirements.txt /tmp/
RUN ~/.local/bin/uv pip install --system --no-cache-dir -r /tmp/requirements.txt

RUN git config --global --add safe.directory /holohub
