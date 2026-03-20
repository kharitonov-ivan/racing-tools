import ast
import sys
from pathlib import Path
from rich.console import Console
from rich.tree import Tree

console = Console(highlight=False)


class ASTExplorer(ast.NodeVisitor):
    def __init__(self, tree: Tree):
        self.tree = tree
        self.stack: list[Tree] = [tree]

    def _calls(self, node: ast.AST) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = getattr(child.func, "id", getattr(child.func, "attr", None))
                if name:
                    calls.append(name)
        return sorted(set(calls))

    def _add(self, label: str):
        self.stack[-1].add(label)

    def visit_ClassDef(self, node: ast.ClassDef):
        self._add(f"[cyan]CLASS[/]: [yellow]{node.name}[/]")
        self.stack.append(self.stack[-1].children[-1])
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        kind = "ASYNC" if isinstance(node, ast.AsyncFunctionDef) else "DEF"
        calls = self._calls(node)
        calls_str = f" -> [{', '.join(calls)}]" if calls else ""
        self._add(f"[cyan]{kind}[/]: [yellow]{node.name}[/]{calls_str}")

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)


def explore(root: Path, exclude: str):
    for py_file in sorted(root.rglob("*.py")):
        rel = str(py_file.relative_to(root))
        if ".venv" in py_file.parts or rel == exclude:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            file_tree = Tree(f"[bold blue]{rel}[/]")
            ASTExplorer(file_tree).visit(tree)
            console.print(file_tree)
        except Exception as e:
            console.print(f"[bold blue]{rel}[/]\n  [red]# ERROR: {e}[/]")


if __name__ == "__main__":
    exclude = str(Path(__file__).relative_to(Path.cwd()))
    explore(Path(sys.argv[1] if len(sys.argv) > 1 else "."), exclude)
