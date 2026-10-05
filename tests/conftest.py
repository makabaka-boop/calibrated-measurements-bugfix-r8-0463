import os
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def demo_data():
    needed = [
        os.path.join(ROOT, "demo", f)
        for f in ("view1.png", "view2.png", "cameras.json")
    ]
    if not all(os.path.exists(p) for p in needed):
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "demo", "make_demo.py")],
            check=True,
            cwd=ROOT,
        )
    return True


@pytest.fixture(scope="session")
def server(demo_data):
    port = 8123
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "server.py"), "--port", str(port)], cwd=ROOT
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("后端服务未能启动")
    yield url
    proc.terminate()
    proc.wait(timeout=5)
