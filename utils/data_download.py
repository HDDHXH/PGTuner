import os
import urllib.request
import ftplib
from tqdm import tqdm

#下载网址维 "http://..."
def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    progress = downloaded / total_size * 100
    print(f"下载进度：{progress:.2f}%")

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
urldict = {'crawl':'http://downloads.zjulearning.org.cn/data/crawl.tar.gz', 'glove':'http://downloads.zjulearning.org.cn/data/glove-100.tar.gz', 'msong':'https://drive.google.com/file/d/1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj/view',
            'uqv':'https://drive.google.com/file/d/1HIdQSKGh7cfC7TnRvrA2dnkHBNkVHGsF/view?usp=sharing', 'paper':'https://drive.google.com/file/d/1t4b93_1Viuudzd5D3I6_9_9Guwm1vmTn/view'}


# filanemes = ['paper', 'msong']
# # filanemes = ['msong', 'paper', 'uqv']
# for fn in tqdm(filanemes, total = len(filanemes)):
#     print(f'正在下载: {fn}')
#     data_path = os.path.join(parent_directory, '{}.tar.gz'.format(fn))
#     url = urldict[fn]
#     urllib.request.urlretrieve(url, data_path, reporthook=report_progress)


#下载网址为 "ftp://..."
def download_file(ftp_address, ftp_path, filename, save_path):
    # 创建FTP连接
    ftp = ftplib.FTP(ftp_address)

    # 登录；这里假设是匿名登录，如果需要用户名和密码，使用 ftp.login(user, passwd)
    ftp.login()

    # 切换到包含文件的目录
    ftp.cwd(ftp_path)

    # 获取文件大小
    total_size = ftp.size(filename)

    # 以二进制模式打开本地文件用于写入
    with open(save_path, 'wb') as local_file:
        # 已下载的数据量
        downloaded = 0

        # 定义回调函数写文件同时更新进度
        def write_to_file(data):
            nonlocal downloaded
            local_file.write(data)
            downloaded += len(data)
            progress = float(downloaded) / total_size * 100
            print(f"下载进度：{progress:.2f}%", end='\r')

        # 使用FTP RETR命令下载文件
        ftp.retrbinary('RETR ' + filename, write_to_file)

    # 关闭FTP连接
    ftp.quit()
    print("\n文件已下载到：", save_path)

#ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz SIFT1b Base  ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz Query


# 使用示例
ftp_address = 'ftp.irisa.fr'
ftp_path = '/local/texmex/corpus/'
# filename1 = 'gist.tar.gz'
# save_path1 = '../Data/gist.tar.gz'
# download_file(ftp_address, ftp_path, filename1, save_path1)

filename2 = 'bigann_base.bvecs.gz'
filename3 = 'bigann_query.bvecs.gz'
save_path2 = '../Data/bigann_base.bvecs.gz'
save_path3 = '../Data/bigann_query.bvecs.gz'
download_file(ftp_address, ftp_path, filename3, save_path3)
download_file(ftp_address, ftp_path, filename2, save_path2)
