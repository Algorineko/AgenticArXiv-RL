#!/usr/bin/env python3
"""
代码文件收集工具
递归获取目录下的代码文件内容，支持忽略规则
扫描指定目录:
python collect_code.py /path/to/your/project
指定输出文件:
python collect_code.py -o context.txt
使用.gitignore规则:
python collect_code.py -i .gitignore
列出默认规则：
python collect_code.py --list-defaults
# 后端
python collect_code.py ./AgenticArxiv -i .gitignore -o collected_code.txt
# 前端
python collect_code.py ./AgenticArxivWeb -i .gitignore -o collected_code_web.txt
"""

import os
import sys
import argparse
from pathlib import Path
import fnmatch
from typing import Set, List, Dict, Optional

class CodeCollector:
    def __init__(self, 
                 root_dir: str = ".",
                 ignore_patterns: Optional[List[str]] = None,
                 include_extensions: Optional[List[str]] = None,
                 max_file_size: int = 1024 * 1024,  # 1MB
                 follow_symlinks: bool = False):
        """
        初始化代码收集器
        
        Args:
            root_dir: 根目录路径
            ignore_patterns: 忽略模式列表
            include_extensions: 包含的文件扩展名
            max_file_size: 最大文件大小（字节）
            follow_symlinks: 是否跟随符号链接
        """
        self.root_dir = Path(root_dir).resolve()
        self.ignore_patterns = ignore_patterns or []
        self.include_extensions = include_extensions or []
        self.max_file_size = max_file_size
        self.follow_symlinks = follow_symlinks
        
        # 默认忽略模式
        self.default_ignore_patterns = [
            # 版本控制
            '.git/', '.svn/', '.hg/', '.bzr/', 
            # 构建产物
            '__pycache__/', '.pyc', '.pyo', '.pyd', 
            '.so', '.dll', '.obj', '.o', '.a', '.lib',
            # 缓存和临时文件
            '.cache/', 'node_modules/', '.next/', '.nuxt/',
            'dist/', 'build/', 'target/', 'out/', '.output/',
            # 环境相关
            '.venv/', 'venv/', 'env/', '.env', '.env.*',
            # IDE相关
            '.idea/', '.vscode/', '.vs/', '*.swp', '*.swo',
            # 打包文件
            '*.zip', '*.tar', '*.gz', '*.7z', '*.rar',
            # 二进制文件
            '*.exe', '*.dmg', '*.pkg', '*.deb', '*.rpm',
            # 媒体文件
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp',
            '*.mp3', '*.mp4', '*.avi', '*.mkv',
            # 文档
            '*.pdf', '*.doc', '*.docx', '*.ppt', '*.pptx',
            # 数据文件
            '*.db', '*.sqlite', '*.dump', '*.csv', '*.log',
        ]
        
        # 合并默认忽略模式
        self.ignore_patterns = self.default_ignore_patterns + self.ignore_patterns
        
        # 常见代码文件扩展名
        self.default_code_extensions = [
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java',
            '.cpp', '.c', '.h', '.hpp', '.cs', '.go',
            '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.html', '.htm', '.css', '.scss', '.sass', '.less',
            '.json', '.yml', '.yaml', '.xml', '.toml', '.ini',
            '.md', '.markdown', '.rst', '.txt', '.sh', '.bash',
            '.sql', '.graphql', '.gql', '.vue', '.svelte',
            '.dart', '.lua', '.perl', '.r', '.m', '.matlab',
            '.tex', '.jl', '.ex', '.exs', '.erl', '.hs', '.lhs',
            '.fs', '.fsx', '.clj', '.cljs', '.scm', '.rkt',
        ]
        
        # 如果没有指定扩展名，使用默认的
        if not self.include_extensions:
            self.include_extensions = self.default_code_extensions
        
        # 已处理的文件路径集合，避免重复处理
        self.processed_paths: Set[Path] = set()
        
    def should_ignore(self, path: Path, is_dir: bool = False) -> bool:
        """
        检查路径是否应该被忽略
        
        Args:
            path: 要检查的路径
            is_dir: 是否是目录
            
        Returns:
            bool: 是否应该忽略
        """
        # 转换为相对路径用于模式匹配
        try:
            rel_path = path.relative_to(self.root_dir)
        except ValueError:
            rel_path = path
            
        # 将Path对象转换为字符串，使用/作为分隔符
        path_str = str(rel_path).replace(os.sep, '/')
        
        # 如果是目录，添加斜杠以便匹配目录模式
        if is_dir:
            path_str += '/'
            
        # 检查所有忽略模式
        for pattern in self.ignore_patterns:
            # 处理目录模式（以/结尾）
            if pattern.endswith('/'):
                if is_dir and fnmatch.fnmatch(path_str, pattern):
                    return True
                # 检查路径是否以该模式开头
                if fnmatch.fnmatch(path_str, pattern[:-1]) or \
                   fnmatch.fnmatch(path_str, pattern[:-1] + '/*'):
                    return True
            else:
                # 常规文件模式匹配
                if fnmatch.fnmatch(path_str, pattern) or \
                   fnmatch.fnmatch(path.name, pattern):
                    return True
        
        return False
    
    def is_text_file(self, file_path: Path) -> bool:
        """
        检查文件是否是文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否是文本文件
        """
        # 检查文件大小
        if file_path.stat().st_size > self.max_file_size:
            return False
            
        # 检查扩展名
        if self.include_extensions:
            ext = file_path.suffix.lower()
            if ext not in self.include_extensions:
                return False
        
        return True
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件内容，如果读取失败则返回None
        """
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'latin-1', 'cp1252', 'gbk', 'gb2312']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，尝试二进制读取并检查是否是文本
            with open(file_path, 'rb') as f:
                content = f.read()
                
            # 简单的文本文件检查（非严谨）
            try:
                return content.decode('utf-8', errors='ignore')
            except:  # noqa: E722
                return None
                
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}", file=sys.stderr)
            return None
    
    def collect_files(self) -> Dict[str, str]:
        """
        收集所有符合条件的文件
        
        Returns:
            Dict[str, str]: 文件路径和内容的字典
        """
        result = {}
        
        if not self.root_dir.exists():
            print(f"错误：目录不存在 {self.root_dir}", file=sys.stderr)
            return result
            
        if not self.root_dir.is_dir():
            print(f"错误：不是目录 {self.root_dir}", file=sys.stderr)
            return result
        
        print(f"正在扫描目录: {self.root_dir}", file=sys.stderr)
        print(f"忽略模式: {self.ignore_patterns[:10]}...", file=sys.stderr)
        print(f"包含扩展名: {self.include_extensions[:10]}...", file=sys.stderr)
        
        # 递归遍历目录
        for root, dirs, files in os.walk(self.root_dir, followlinks=self.follow_symlinks):
            root_path = Path(root)
            
            # 过滤要遍历的目录
            dirs[:] = [
                d for d in dirs 
                if not self.should_ignore(root_path / d, is_dir=True)
            ]
            
            # 处理文件
            for file in files:
                file_path = root_path / file
                
                # 避免重复处理
                if file_path in self.processed_paths:
                    continue
                    
                self.processed_paths.add(file_path)
                
                # 检查是否应该忽略
                if self.should_ignore(file_path):
                    continue
                    
                # 检查是否是文本文件
                if not self.is_text_file(file_path):
                    continue
                
                # 读取文件内容
                content = self.read_file_content(file_path)
                if content is not None:
                    # 使用相对路径作为键
                    rel_path = file_path.relative_to(self.root_dir)
                    result[str(rel_path)] = content
                    print(f"✓ 已添加: {rel_path}", file=sys.stderr)
        
        print(f"\n总计收集了 {len(result)} 个文件", file=sys.stderr)
        return result
    
    def save_to_file(self, output_file: str = "project_context.txt"):
        """
        将收集到的代码保存到文件
        
        Args:
            output_file: 输出文件路径
        """
        files_content = self.collect_files()
        
        if not files_content:
            print("没有找到符合条件的文件", file=sys.stderr)
            return
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for file_path, content in files_content.items():
                    # 写入分隔符和文件路径
                    f.write(f"\n{'='*80}\n")
                    f.write(f"文件: {file_path}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(content)
                    f.write("\n")
                    
            print(f"\n已保存到: {output_file}", file=sys.stderr)
            print(f"总文件数: {len(files_content)}", file=sys.stderr)
            
        except Exception as e:
            print(f"保存文件失败: {e}", file=sys.stderr)

def load_ignore_patterns(ignore_file: str = ".gitignore") -> List[str]:
    """
    从文件加载忽略模式
    
    Args:
        ignore_file: 忽略文件路径
        
    Returns:
        List[str]: 忽略模式列表
    """
    patterns = []
    
    if os.path.exists(ignore_file):
        try:
            with open(ignore_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue
                        
                    patterns.append(line)
                    
            print(f"从 {ignore_file} 加载了 {len(patterns)} 个忽略规则", file=sys.stderr)
        except Exception as e:
            print(f"读取忽略文件失败 {ignore_file}: {e}", file=sys.stderr)
    
    return patterns

def main():
    parser = argparse.ArgumentParser(
        description="递归收集目录下的代码文件内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 收集当前目录
  %(prog)s /path/to/project         # 收集指定目录
  %(prog)s -i .gitignore            # 使用.gitignore规则
  %(prog)s -x ".py .js"             # 只收集.py和.js文件
  %(prog)s -o context.txt           # 保存到context.txt
  %(prog)s -e ".log .tmp"           # 额外忽略.log和.tmp文件
        """
    )
    
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="要扫描的目录（默认：当前目录）"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="project_context.txt",
        help="输出文件名（默认：project_context.txt）"
    )
    
    parser.add_argument(
        "-i", "--ignore-file",
        help="忽略规则文件（如.gitignore）"
    )
    
    parser.add_argument(
        "-x", "--extensions",
        help="包含的文件扩展名，用空格分隔（如：'.py .js .html'）"
    )
    
    parser.add_argument(
        "-e", "--extra-ignore",
        help="额外的忽略模式，用空格分隔"
    )
    
    parser.add_argument(
        "-m", "--max-size",
        type=int,
        default=1024 * 1024,
        help="最大文件大小（字节，默认：1MB）"
    )
    
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="跟随符号链接"
    )
    
    parser.add_argument(
        "--no-default-ignore",
        action="store_true",
        help="不使用默认忽略规则"
    )
    
    parser.add_argument(
        "--list-defaults",
        action="store_true",
        help="列出默认的忽略规则和扩展名"
    )
    
    args = parser.parse_args()
    
    # 如果请求列出默认值
    if args.list_defaults:
        collector = CodeCollector()
        print("默认忽略规则:")
        for pattern in collector.default_ignore_patterns:
            print(f"  {pattern}")
        print("\n默认代码文件扩展名:")
        for ext in collector.default_code_extensions:
            print(f"  {ext}")
        return
    
    # 准备忽略模式
    ignore_patterns = []
    
    # 加载忽略文件
    if args.ignore_file:
        ignore_patterns.extend(load_ignore_patterns(args.ignore_file))
    else:
        # 自动查找.gitignore
        gitignore_path = os.path.join(args.directory, ".gitignore")
        if os.path.exists(gitignore_path):
            ignore_patterns.extend(load_ignore_patterns(gitignore_path))
    
    # 添加额外忽略模式
    if args.extra_ignore:
        ignore_patterns.extend(args.extra_ignore.split())
    
    # 准备扩展名列表
    include_extensions = None
    if args.extensions:
        include_extensions = args.extensions.split()
    
    # 如果指定了不使用默认忽略规则
    if args.no_default_ignore:
        # 只使用用户提供的忽略规则
        collector = CodeCollector(
            root_dir=args.directory,
            ignore_patterns=ignore_patterns,
            include_extensions=include_extensions,
            max_file_size=args.max_size,
            follow_symlinks=args.follow_symlinks
        )
        # 清空默认忽略规则
        collector.default_ignore_patterns = []
        collector.ignore_patterns = ignore_patterns
    else:
        # 使用所有规则
        collector = CodeCollector(
            root_dir=args.directory,
            ignore_patterns=ignore_patterns,
            include_extensions=include_extensions,
            max_file_size=args.max_size,
            follow_symlinks=args.follow_symlinks
        )
    
    # 收集并保存文件
    collector.save_to_file(args.output)

if __name__ == "__main__":
    main()