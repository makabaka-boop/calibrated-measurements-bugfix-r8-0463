"""双视角教学量测台后端：静态页面 + /api/triangulate。

无状态服务：不做任何持久化，不做自标定（K/R/t 只校验、不修改）。
"""

import argparse
import os

from flask import Flask, jsonify, request, send_from_directory

import triangulation as tri

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/demo/<path:filename>")
def demo_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "demo"), filename)


@app.post("/api/triangulate")
def api_triangulate():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error="请求体必须是 JSON 对象"), 400
    try:
        cam1 = {"K": data["K1"], "R": data["R1"], "t": data["t1"]}
        cam2 = {"K": data["K2"], "R": data["R2"], "t": data["t2"]}
        pairs = data["pairs"]
        max_px_err = float(data["max_px_err"])
        sigma = float(data["sigma"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(ok=False, error=f"请求字段缺失或非法：{e}"), 400

    try:
        results = tri.triangulate_all(cam1, cam2, pairs, max_px_err, sigma)
    except tri.TriangulationRejected as e:
        return jsonify(ok=False, code=e.code, error=e.message), 400
    return jsonify(ok=True, results=results)


@app.post("/api/measurements")
def api_measurements():
    from measurements import MeasurementFailure, measure_request

    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify(ok=False, code="INVALID_REQUEST", error="请求体必须是 JSON 对象"), 400
    try:
        return jsonify(measure_request(payload))
    except MeasurementFailure as exc:
        # 整次量测失败：稳定 code + 说明 + 同次三角化证据（保留原始拒绝原因）
        return (
            jsonify(
                ok=False,
                code=exc.code,
                error=exc.message,
                results=exc.results,
            ),
            400,
        )
    except tri.TriangulationRejected as exc:
        return jsonify(ok=False, code=exc.code, error=exc.message), 400
    except KeyError as exc:
        return jsonify(
            ok=False, code="INVALID_REQUEST", error=f"缺少字段：{exc.args[0]}"
        ), 400
    except (ValueError, TypeError) as exc:
        return jsonify(ok=False, code="INVALID_REQUEST", error=str(exc)), 400


@app.get("/measurements")
def measurement_page():
    return send_from_directory(app.static_folder, "measurements.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
