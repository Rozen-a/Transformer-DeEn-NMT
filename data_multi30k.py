import os
from sacremoses import MosesTokenizer
from pathlib import Path
import argparse


def moses_cut(in_file, out_file, lang):
    """
    使用MosesTokenizer进行分词

    Args:
        in_file: 输入文件
        out_file: 输出文件
        lang: 语言类型
    """
    mt = MosesTokenizer(lang=lang)  # 初始化分词器，lang是语言类型，如de,en，是sacremoses库中的语言类型，不能乱写
    out_f = open(out_file, "w", encoding="utf8") #新建并打开一个文件
    with open(in_file, "r", encoding="utf8") as f:
        for line in f.readlines():  # 每读取一行，进行分词，并写入一行到新的文件中
            line = line.strip()
            if not line:
                continue
            cut_line = mt.tokenize(line, return_str=True)  # 分词
            out_f.write(cut_line.lower() + "\n")  # 变为小写，并写入文件
    out_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建解析器

    # 添加参数
    parser.add_argument(
        "-p",             # 短选项（Short option），方便在命令行快速输入，如 -p ./data
        "--pair_dir",     # 长选项（Long option），代码内部调用的变量名将被解析为 args.pair_dir
        default=None,     # 默认值。如果用户执行脚本时未提供此参数，则值为 None
        type=str,         # 类型约束。argparse 会自动将终端传入的字符串转换为指定类型
        help="The directory which contains language pair files." # 帮助文档
    )
    parser.add_argument(
        "-d",
        "--dest_dir",
        default=None,
        type=str,
        help="The destination directory to save processed train, dev and test file.",
    )
    parser.add_argument("--src_lang", default="de", type=str, help="source language")
    parser.add_argument("--trg_lang", default="en", type=str, help="target language")

    args = parser.parse_args()  # 解析参数，args是一个列表，包含了传递的参数值
    if not args.pair_dir:  # 如果不传参，就抛异常
        raise ValueError("Please specify --pair_dir")
    # 判断args.dest_dir是否存在,不存在就创建
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    local_data_path = Path(args.pair_dir)  # 获取本地原始数据路径
    data_dir = Path(args.dest_dir)  # 获取处理后的数据保存路径

    # 分词
    for mode in ["train", "val", "test"]:
        moses_cut(
            local_data_path / f"{mode}.{args.src_lang}",  # 读取源语言文件
            data_dir / f"{mode}_src.cut.txt",
            lang=args.src_lang,
        )
        print(f"[{mode}] 源语言文本分词完成")
        moses_cut(
            local_data_path / f"{mode}.{args.trg_lang}",  # 读取目标语言文件
            data_dir / f"{mode}_trg.cut.txt",
            lang=args.trg_lang,
        )
        print(f"[{mode}] 目标语言文本分词完成")

