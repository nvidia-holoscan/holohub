# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the chat servers' actual HTTP error responses without model inference."""

import importlib.util
import queue
import sys
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
import werkzeug.serving


@pytest.mark.parametrize("application", ["vila_live", "ehr_query_llm/lmm"])
@pytest.mark.parametrize("flask_debug", ["0", "1"])
def test_http_errors_do_not_expose_debugger(application, flask_debug, monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", flask_debug)
    if application == "ehr_query_llm/lmm":
        # Only the unused WebServerOp wrapper needs these SDK types. Keep the
        # HTTP and WebSocket servers real without importing the GPU runtime.
        core = ModuleType("holoscan.core")
        core.Operator = object
        core.OperatorSpec = object
        holoscan = ModuleType("holoscan")
        holoscan.core = core
        monkeypatch.setitem(sys.modules, "holoscan", holoscan)
        monkeypatch.setitem(sys.modules, "holoscan.core", core)
    path = Path(__file__).resolve().parents[2] / "applications" / application / "webserver.py"
    module_name = "test_webserver_" + application.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    server = module.Webserver(web_server="127.0.0.1", web_port=0, ws_port=0)

    def fail():
        raise RuntimeError("private-error-detail-must-not-reach-client")

    server.app.add_url_rule("/failure", view_func=fail)
    server.app.add_url_rule("/healthy", view_func=lambda: "healthy")
    ready = queue.Queue()
    make_server = werkzeug.serving.make_server

    def capture_server(*args, **kwargs):
        http_server = make_server(*args, **kwargs)
        ready.put(http_server)
        return http_server

    monkeypatch.setattr(werkzeug.serving, "make_server", capture_server)
    server.start()
    http_server = ready.get(timeout=10)
    base_url = f"http://127.0.0.1:{http_server.server_port}"
    try:
        with urlopen(base_url + "/healthy", timeout=5) as response:
            assert response.read() == b"healthy"
        with pytest.raises(HTTPError) as error:
            urlopen(base_url + "/failure", timeout=5)
        with error.value as response:
            assert response.code == 500
            body = response.read()
        assert b"private-error-detail-must-not-reach-client" not in body
        assert b"Werkzeug Debugger" not in body
        with pytest.raises(HTTPError) as error:
            urlopen(base_url + "/console", timeout=5)
        error.value.close()
        assert error.value.code == 404
    finally:
        http_server.shutdown()
        server.join(timeout=5)
        server.ws_server.shutdown()
        server.ws_thread.join(timeout=5)
