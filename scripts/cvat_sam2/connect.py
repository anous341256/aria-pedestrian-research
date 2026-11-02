"""Launch the unmodified official CVAT tracker; never persist the access token."""
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SERVER = "https://app.cvat.ai"
NAME = "Aria SAM2.1 Large - official local GPU"
MODEL = "facebook/sam2.1-hiera-large"
ADAPTER = ROOT / "tools/cvat-models-official/tracker/sam2/func.py"
CLI = "/opt/cvat-sam2/bin/cvat-cli"
CACHE = "/opt/cvat-sam2-cache/huggingface"


def api_get(path, token):
    request = urllib.request.Request(
        SERVER + path, headers={"Authorization": "Bearer " + token}
    )
    # Do not follow redirects with an Authorization header.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        return json.load(response)


def find_function(token, username):
    query = {"and": [
        {"==": [{"var": "name"}, NAME]},
        {"==": [{"var": "owner"}, username]},
    ]}
    page = api_get("/api/functions?" + urllib.parse.urlencode({
        "filter": json.dumps(query), "page_size": 100, "org": "",
    }), token)
    matches = page.get("results", [])
    if len(matches) > 1:
        raise RuntimeError("找到多个同名服务，请先核对 CVAT Models，不自动选择。")
    if not matches:
        return None
    function = matches[0]
    if function.get("visibility") != "private" or function.get("kind") != "tracker":
        raise RuntimeError("同名服务不是私有 Tracker，已停止，未修改它。")
    if function.get("provider") != "native":
        raise RuntimeError("同名服务不是 native agent，已停止，未修改它。")
    return int(function["id"])


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("请双击项目根目录 Open_CVAT_SAM2_Tracker.cmd，在交互窗口输入令牌。")
    os.environ["HF_HOME"] = CACHE
    os.environ["HF_HUB_OFFLINE"] = "1"
    lock_path = Path("/opt/cvat-sam2-cache/agent.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("另一个 SAM2 Tracker 启动窗口仍在运行，请使用原窗口。")
        print("官方 SAM 2.1 Large / CVAT 私有 Tracker / 本机 GPU\n")
        print("令牌入口：https://app.cvat.ai/profile#security")
        print("请创建有效的、非只读 CVAT Personal Access Token；建议设置短有效期。")
        print("令牌只在本次运行内存中使用，不保存到文件，不要发到聊天里。")
        print("本程序只注册/运行追踪服务，不上传视频，不自动执行或覆盖标注。\n")
        token = getpass.getpass("粘贴 CVAT Access Token（不显示字符），然后回车：").strip()
        if not token:
            print("未输入令牌，已退出。")
            return 0
        user = api_get("/api/users/self", token)
        print("已登录：", user["username"], flush=True)
        function_id = find_function(token, user["username"])
        child_env = dict(os.environ, CVAT_ACCESS_TOKEN=token)
        args = [CLI, "--server-host", SERVER, "--organization", ""]
        params = ["--function-file", str(ADAPTER), "-p", "model_id=str:" + MODEL,
                  "-p", "device=str:cuda"]
        if function_id is None:
            print("正在注册私有模型入口（不会修改任务标注）…", flush=True)
            completed = subprocess.run(
                args + ["function", "create-native", NAME, "--visibility", "private"] + params,
                env=child_env, text=True, stdout=subprocess.PIPE, check=True,
            )
            ids = [line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit()]
            if len(ids) != 1:
                raise RuntimeError("注册返回值无法识别。请重启：会先查找已创建的入口，不盲目重复创建。")
            function_id = int(ids[0])
        else:
            print(f"复用私有模型入口 #{function_id}。", flush=True)
        # Non-secret registration metadata only. Never write token or environment.
        state_dir = ROOT / "outputs/cvat_sam2_setup"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "registration.json").write_text(json.dumps({
            "server": SERVER, "username": user["username"], "function_id": function_id,
            "name": NAME, "model": MODEL, "visibility": "private",
        }, indent=2), encoding="utf-8")
        print("\n服务即将启动；请保持此窗口打开，刷新 CVAT 的 Models 页面。", flush=True)
        print("名称：" + NAME, flush=True)
        print("停止：Ctrl+C。停止后不能继续追踪；下次启动需重新输入令牌。\n", flush=True)
        return subprocess.call(args + ["function", "run-agent", str(function_id)] + params,
                               env=child_env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n服务已停止；CVAT 中已保存的标注未被删除。")
        sys.exit(130)
    except urllib.error.HTTPError as exc:
        print(f"CVAT 返回 HTTP {exc.code}。401/403 请检查令牌、有效期及访问权限；未自动重试。")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"官方客户端退出，代码 {exc.returncode}。请查看上方错误；不要分享令牌。")
        sys.exit(1)
    except Exception as exc:
        print(f"启动未完成：{exc}")
        sys.exit(1)
