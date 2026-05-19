"""文件监听守护：自动触发 compaction。

使用 watchdog 监听 opt_memory/dialog/ 目录变化。
当检测到新的对话 JSONL 文件或文件增长超过阈值时，自动调用 compact()。
"""

import time
from pathlib import Path

from .common import get_opt_dir


def watch(
    *,
    operator: str | None = None,
    threshold_mb: float = 5.0,
    cooldown_seconds: int = 300,
    working_dir: str = ".",
) -> None:
    """监听 opt_memory/ 目录，触发自动压缩。

    Args:
        operator: 指定监听的算子，不指定则监听所有。
        threshold_mb: 对话文件大小阈值（MB），超过此值触发压缩。
        cooldown_seconds: 两次压缩之间的最小间隔（秒）。

    保持前台运行，Ctrl+C 退出。
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("错误: watchdog 未安装。请运行: uv sync")
        return

    opt_dir = get_opt_dir(working_dir)
    dialog_dir = opt_dir / "dialog"
    dialog_dir.mkdir(parents=True, exist_ok=True)

    processed_files: dict[str, float] = {}  # 已处理文件及其大小
    last_compact_time = 0.0

    threshold_bytes = threshold_mb * 1024 * 1024

    class DialogHandler(FileSystemEventHandler):
        def on_modified(self, event):
            nonlocal last_compact_time
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix != ".jsonl":
                return

            now = time.time()
            if now - last_compact_time < cooldown_seconds:
                return

            try:
                size = path.stat().st_size
            except OSError:
                return

            prev_size = processed_files.get(str(path), 0)
            processed_files[str(path)] = size

            if size > threshold_bytes and size > prev_size:
                op = operator or "all"
                print(f"[watch] 对话文件 {path.name} 大小 {size / 1024 / 1024:.1f}MB，"
                      f"触发压缩 (operator={op})")
                last_compact_time = now

                from .compact import compact
                result = compact(
                    operator=op,
                    dialog=str(path),
                    working_dir=working_dir,
                )
                if "error" in result:
                    print(f"[watch] 压缩失败: {result['error']}")
                else:
                    print(f"[watch] 压缩完成 → {result.get('path', '')}")

        def on_created(self, event):
            self.on_modified(event)

    observer = Observer()
    handler = DialogHandler()
    observer.schedule(handler, str(dialog_dir), recursive=False)
    observer.start()

    print(f"[watch] 开始监听 {dialog_dir}（阈值 {threshold_mb}MB，"
          f"冷却 {cooldown_seconds}s）")
    print("[watch] 按 Ctrl+C 退出")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[watch] 已停止")
    observer.join()
