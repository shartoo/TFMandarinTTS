# -*- coding:utf-8 -*-

import logging
import os

def get_logger(log_name,log_save_file="mandarin_tts.log"):
    log_save_file = os.path.join("../data/log",log_save_file)
    if not (os.path.exists(log_save_file)):  # 检查文件是否已经存在
        fo = open(log_save_file, "w")
        fo.close()
    fsize = os.path.getsize(log_save_file)
    fsize = fsize / float(1024 * 1024)
    if fsize > 4:
        os.remove(log_save_file)
        print("log file is remove log file..",log_save_file)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    # file log handler
    fh = logging.FileHandler(log_save_file)
    fh.setLevel(logging.DEBUG)
    # console hander
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s %(filename)s\t[line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
