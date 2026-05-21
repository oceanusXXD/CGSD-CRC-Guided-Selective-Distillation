#!/usr/bin/env python
"""检查保留 Python 代码的 AST 和内部导入关系。

这个脚本只做静态检查，不导入业务模块，适合在没有模型依赖或 GPU 的环境
里快速确认代码树是否还能被解析、内部模块是否缺失、以及是否出现循环引用。
"""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AstIntegrityReport:
    syntax_errors: list[str]
    missing_internal_imports: list[str]
    import_cycles: list[list[str]]
    module_count: int
    edge_count: int


def _python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
            continue
        if path.is_dir():
            files.extend(
                item
                for item in path.rglob("*.py")
                if "__pycache__" not in item.parts
            )
    return sorted({item.resolve() for item in files})


def _module_name(path: Path, project_root: Path) -> str:
    relative = path.relative_to(project_root).with_suffix("")
    if relative.name == "__init__":
        relative = relative.parent
    return ".".join(relative.parts)


def _resolve_relative_import(module_name: str, level: int, imported_module: str | None) -> str:
    parts = module_name.split(".")
    package_parts = parts[:-1]
    if level > 1:
        package_parts = package_parts[: -(level - 1)]
    if imported_module:
        package_parts.extend(imported_module.split("."))
    return ".".join(part for part in package_parts if part)


def _import_targets(module_name: str, tree: ast.AST) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                resolved = _resolve_relative_import(module_name, int(node.level), node.module)
                if resolved:
                    targets.append(resolved)
            elif node.module:
                targets.append(node.module)
    return targets


def _is_known_package(target: str, modules: set[str]) -> bool:
    return any(module == target or module.startswith(f"{target}.") for module in modules)


def _is_internal(target: str, internal_roots: set[str]) -> bool:
    first = target.split(".", 1)[0]
    return first in internal_roots


def _edge_target(target: str, modules: set[str]) -> str | None:
    if target in modules:
        return target
    candidates = [module for module in modules if module.startswith(f"{target}.")]
    if candidates:
        return sorted(candidates)[0]
    return None


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            start = visiting.index(node)
            cycles.append(visiting[start:] + [node])
            return
        if node in visited:
            return
        visiting.append(node)
        for target in sorted(graph.get(node, ())):
            visit(target)
        visiting.pop()
        visited.add(node)

    for node in sorted(graph):
        visit(node)
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cycle in cycles:
        key = tuple(cycle)
        if key not in seen:
            deduped.append(cycle)
            seen.add(key)
    return deduped


def analyze_paths(paths: list[Path]) -> AstIntegrityReport:
    resolved_paths = [path.resolve() for path in paths]
    files = _python_files(resolved_paths)
    if not files:
        return AstIntegrityReport([], [], [], 0, 0)

    project_root = Path(os.path.commonpath([str(path if path.is_dir() else path.parent) for path in resolved_paths]))
    modules_by_path = {path: _module_name(path, project_root) for path in files}
    modules = set(modules_by_path.values())
    internal_roots = {module.split(".", 1)[0] for module in modules}
    syntax_errors: list[str] = []
    missing_internal_imports: list[str] = []
    graph: dict[str, set[str]] = {module: set() for module in modules}

    for path, module_name in modules_by_path.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(f"{path}:{exc.lineno}:{exc.offset}: {exc.msg}")
            continue
        for target in _import_targets(module_name, tree):
            if not _is_internal(target, internal_roots):
                continue
            if not _is_known_package(target, modules):
                missing_internal_imports.append(f"{module_name} -> {target}")
                continue
            edge = _edge_target(target, modules)
            if edge and edge != module_name:
                graph[module_name].add(edge)

    return AstIntegrityReport(
        syntax_errors=sorted(syntax_errors),
        missing_internal_imports=sorted(set(missing_internal_imports)),
        import_cycles=_find_cycles(graph),
        module_count=len(modules),
        edge_count=sum(len(edges) for edges in graph.values()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src", "scripts"],
        help="要检查的文件或目录，默认检查核心 Python 代码目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_paths([Path(path) for path in args.paths])
    if report.syntax_errors or report.missing_internal_imports or report.import_cycles:
        if report.syntax_errors:
            print("SYNTAX ERRORS:")
            for item in report.syntax_errors:
                print(f"  {item}")
        if report.missing_internal_imports:
            print("MISSING INTERNAL IMPORTS:")
            for item in report.missing_internal_imports:
                print(f"  {item}")
        if report.import_cycles:
            print("IMPORT CYCLES:")
            for cycle in report.import_cycles:
                print("  " + " -> ".join(cycle))
        raise SystemExit(1)
    print(
        f"AST/import graph OK: modules={report.module_count}, edges={report.edge_count}"
    )


if __name__ == "__main__":
    main()
