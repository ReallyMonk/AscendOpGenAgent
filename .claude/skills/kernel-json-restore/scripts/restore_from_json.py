#!/usr/bin/env python3
"""
从 JSON 文件还原 AscendC Kernel 算子项目
"""

import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def get_file_md5(filepath: Path) -> str:
    """计算文件 MD5"""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_file_content(file_path: Path, expected_content: str) -> bool:
    """验证文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            actual_content = f.read()
        return actual_content == expected_content
    except Exception:
        return False


def restore_operator(
    operator_name: str,
    json_dir: Path,
    output_dir: Path,
    verify_md5: bool = True
) -> Dict[str, Any]:
    """
    从 JSON 还原单个算子

    Args:
        operator_name: 算子名称
        json_dir: JSON 文件目录
        output_dir: 输出目录
        verify_md5: 是否验证 MD5

    Returns:
        还原结果统计
    """
    json_file = json_dir / f"{operator_name}.json"

    if not json_file.exists():
        # 尝试从 all_operators.json 中查找
        all_json_file = json_dir / "all_operators.json"
        if all_json_file.exists():
            with open(all_json_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            for op_data in all_data.get("operators", []):
                if op_data["operator_name"] == operator_name:
                    json_data = op_data
                    break
            else:
                return {"success": False, "error": f"算子 {operator_name} 不存在"}
        else:
            return {"success": False, "error": f"JSON 文件不存在: {json_file}"}
    else:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

    # 创建输出目录
    operator_output_dir = output_dir / operator_name / "kernel"
    operator_output_dir.mkdir(parents=True, exist_ok=True)

    # 还原文件
    restored_files = []
    failed_files = []

    for file_name, file_info in json_data["files"].items():
        file_path = operator_output_dir / file_name

        try:
            # 确保父目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_info["content"])

            # MD5 验证
            if verify_md5:
                actual_md5 = get_file_md5(file_path)
                expected_md5 = file_info["md5"]

                if actual_md5 != expected_md5:
                    failed_files.append({
                        "file": file_name,
                        "error": f"MD5 不匹配: 期望 {expected_md5}, 实际 {actual_md5}"
                    })
                else:
                    restored_files.append(file_name)
            else:
                restored_files.append(file_name)

        except Exception as e:
            failed_files.append({
                "file": file_name,
                "error": str(e)
            })

    # 统计结果
    result = {
        "success": len(failed_files) == 0,
        "operator_name": operator_name,
        "output_dir": str(operator_output_dir),
        "restored_files": len(restored_files),
        "failed_files": len(failed_files),
        "total_files": json_data["metadata"]["total_files"],
        "details": {
            "restored": restored_files,
            "failed": failed_files
        }
    }

    return result


def restore_all_operators(
    json_dir: Path,
    output_dir: Path,
    verify_md5: bool = True
) -> Dict[str, Any]:
    """
    还原所有算子

    Args:
        json_dir: JSON 文件目录
        output_dir: 输出目录
        verify_md5: 是否验证 MD5

    Returns:
        批量还原结果统计
    """
    summary_file = json_dir / "summary.json"

    if not summary_file.exists():
        return {"success": False, "error": "summary.json 不存在"}

    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    results = {
        "total_operators": summary["total_operators"],
        "success_count": 0,
        "failed_count": 0,
        "operators": []
    }

    print(f"📊 开始还原 {summary['total_operators']} 个算子...\n")

    for i, op_info in enumerate(summary["operators"], 1):
        op_name = op_info["name"]

        print(f"[{i}/{summary['total_operators']}] 还原 {op_name}...", end=" ")

        result = restore_operator(op_name, json_dir, output_dir, verify_md5)

        if result["success"]:
            print(f"✅ ({result['restored_files']} 文件)")
            results["success_count"] += 1
        else:
            print(f"❌ {result['error']}")
            results["failed_count"] += 1

        results["operators"].append(result)

    return results


def list_available_operators(json_dir: Path) -> None:
    """列出所有可用的算子"""
    summary_file = json_dir / "summary.json"

    if not summary_file.exists():
        print("❌ summary.json 不存在")
        return

    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    print(f"📋 可用算子列表 (共 {summary['total_operators']} 个):\n")

    for i, op_info in enumerate(summary["operators"], 1):
        print(f"  {i:2d}. {op_info['name']:<40} ({op_info['files']} 文件, {op_info['size_kb']:.1f} KB)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="从 JSON 文件还原 AscendC Kernel 算子项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 还原单个算子
  python restore_from_json.py layer_norm_v3 -o /tmp/output

  # 还原所有算子
  python restore_from_json.py --all -o /tmp/output

  # 列出所有可用算子
  python restore_from_json.py --list

  # 不验证 MD5
  python restore_from_json.py layer_norm_v3 -o /tmp/output --no-verify
        """
    )

    parser.add_argument(
        "operator_name",
        nargs="?",
        help="算子名称（如 layer_norm_v3）"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./restored_operators"),
        help="输出目录（默认: ./restored_operators）"
    )

    parser.add_argument(
        "-j", "--json-dir",
        type=Path,
        default=Path("/home/hrl/AscendOpGenAgent/litebench/baseline_json"),
        help="JSON 文件目录"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="还原所有算子"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的算子"
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过 MD5 验证"
    )

    args = parser.parse_args()

    # 列出所有算子
    if args.list:
        list_available_operators(args.json_dir)
        return

    # 还原所有算子
    if args.all:
        results = restore_all_operators(
            args.json_dir,
            args.output,
            verify_md5=not args.no_verify
        )

        print("\n" + "=" * 60)
        print("📊 批量还原完成！")
        print(f"✅ 成功: {results['success_count']}")
        print(f"❌ 失败: {results['failed_count']}")
        print(f"📁 输出目录: {args.output}")
        print("=" * 60)
        return

    # 还原单个算子
    if not args.operator_name:
        parser.print_help()
        print("\n❌ 请提供算子名称，或使用 --all 还原所有算子")
        sys.exit(1)

    print(f"🔄 开始还原算子: {args.operator_name}")
    print(f"📂 JSON 目录: {args.json_dir}")
    print(f"📤 输出目录: {args.output}")
    print()

    result = restore_operator(
        args.operator_name,
        args.json_dir,
        args.output,
        verify_md5=not args.no_verify
    )

    if result["success"]:
        print("\n✅ 还原成功！")
        print(f"📁 输出目录: {result['output_dir']}")
        print(f"📄 还原文件数: {result['restored_files']}/{result['total_files']}")
        print("\n📋 还原的文件:")
        for file_name in result["details"]["restored"]:
            print(f"  - {file_name}")

        if result["details"]["failed"]:
            print("\n⚠️ 验证失败的文件:")
            for fail_info in result["details"]["failed"]:
                print(f"  - {fail_info['file']}: {fail_info['error']}")
    else:
        print(f"\n❌ 还原失败: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
