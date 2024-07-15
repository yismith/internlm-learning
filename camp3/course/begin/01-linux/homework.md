
# [Linux基础知识](https://github.com/InternLM/Tutorial/blob/camp3/docs/L0/Linux/readme.md)

## 闯关任务
> 完成SSH连接与端口映射并运行`hello_world.py`

SSH连接信息
![01-ssh-info](../../../assets/image/begin/01/01-ssh-info.png)

进行SSH连接
![01-ssh-connection](../../../assets/image/begin/01/01-ssh-connection.png)


自定义服务信息
![01-port-forwarding-info](../../../assets/image/begin/01/01-port-forwarding-info.png)

端口转发
![01-port-forwarding](../../../assets/image/begin/01/01-port-forwarding.png)


创建linux任务目录
```shell
mkdir -p /root/camp3/l0/linux
cd /root/camp3/l0/linux/
```

编写hello world脚本
```shell
vi hello_world.py
```

脚本内容
```python
import socket
import re
import gradio as gr
 
# 获取主机名
def get_hostname():
    hostname = socket.gethostname()
    match = re.search(r'-(\d+)$', hostname)
    name = match.group(1)
    
    return name
 
# 创建 Gradio 界面
with gr.Blocks(gr.themes.Soft()) as demo:
    html_code = f"""
            <p align="center">
            <a href="https://intern-ai.org.cn/home">
                <img src="https://intern-ai.org.cn/assets/headerLogo-4ea34f23.svg" alt="Logo" width="20%" style="border-radius: 5px;">
            </a>
            </p>
            <h1 style="text-align: center;">☁️ Welcome {get_hostname()} user, welcome to the ShuSheng LLM Practical Camp Course!</h1>
            <h2 style="text-align: center;">😀 Let’s go on a journey through ShuSheng Island together.</h2>
            <p align="center">
                <a href="https://github.com/InternLM/Tutorial/blob/camp3">
                    <img src="https://oss.lingkongstudy.com.cn/blog/202406301604074.jpg" alt="Logo" width="20%" style="border-radius: 5px;">
                </a>
            </p>

            """
    gr.Markdown(html_code)

demo.launch()
```

安装gradio依赖
```shell
pip install gradio==4.29.0
```

运行脚本
```shell
python hello_world.py
```

浏览器访问结果
![01-hello-world-result](../../../assets/image/begin/01/01-hello-world-result.png)


历史操作
```shell
    1  du -sh ./
    2  git -version
    3  git --version
    4  git lfs --version
    5  # install git
    6  sudo apt-get update
    7  sudo apt-get install git
    8  # install git lfs
    9  sudo apt-get update
   10  sudo apt-get install git-lfs
   11  # use git install lfs
   12  mkdir -p /root/camp3/l0/linux
   13  cd /root/camp3/l0/linux/
   14  vi hello_world.py
   15  pip install gradio==4.29.0
   16  python hello_world.py
   17  history
```
![01-hello-world-history-operation](../../../assets/image/begin/01/01-hello-world-history-operation.png)


## 可选任务 1
> 将Linux基础命令在开发机上完成一遍 

### 文件管理
1. touch
2. mkdir 
3. cd 
4. pwd 
5. cat 
6. vi or vim
7. cp 与 ln
8. mv 与 rm
9. find
10. ls
11. sed

### 进程管理
1. ps
2. top
3. pstree
4. pgrep
5. nice
6. jobs
7. bg 与 fg
8. kill

### nvidia-smi相关
1. nvidia-smi
2. nvidia-smi -l 1
3. nvidia-smi pmon
4. nvidia-smi --id=0 --ex_pid=12345
5. nvidia-smi -pm 1
6. nvidia-smi -i 0 -pm 1
7. nvidia-smi --id=0 -r
8. nvidia-smi -h

### tmux
1. tmux --help

历史操作
```shell
    1  vi ~/.bash_history 
    2  touch demo.py
    3  mkdir test
    4  cd test/
    5  cd ..
    6  pwd
    7  cat --help
    8  cat -n 5 hello_world.py 
    9  vi demo.py 
   10  cat demo.py 
   11  cp -vrf demo.py demo_copy.py 
   12  ln -s demo.py demo_link.py
   13  ll
   14  cp -vrf demo.py demo_tmp.py
   15  mv demo_tmp.py demo_tmp.py.bak
   16  rm -vrf demo_tmp.py
   17  find ./ -name "*.py"
   18  ls -a
   19  ls -lh
   20  echo 'InternLM' > file
   21  cat file 
   22  sed -e 's/InternLM/InternLM yyds/g' file 
   23  sed -n '/InternLM/p' file
   24  cat file 
   25  pip list|grep os
   26  ps aux
   27  top
   28  pstree
   29  apt-get update
   30  apt-get install pstree
   31  pstree
   32  apt-get install psmisc
   33  pstree
   34  pgrep -u root
   35  nice -h
   36  nice --help
   37  jobs
   38  bg --help
   39  fg --help
   40  kill --help
   41  nvidia-smi 
   42  nvidia-smi -l 1
   43  nvidia-smi -h
   44  nvidia-smi pmon
   45  nvidia-smi -h
   46  lsb_release -a
   47  apt-get install tmux
   48  tmux --help
   49  history
```
![01-task1-history-operation](../../../assets/image/begin/01/01-task1-history-operation.png)

![01-task1-history-operation-detail](../../../assets/image/begin/01/01-task1-history-operation-detail.png)



## 可选任务 2
> 使用 VSCODE 远程连接开发机并创建一个conda环境

安装ssh扩展

![01-task2-install-ssh-extension](../../../assets/image/begin/01/01-task2-install-ssh-extension.png)



连接

![02-task2-start-ssh](../../../assets/image/begin/01/01-task2-start-ssh.png)

打开路径

![02-task2-workspace-01](../../../assets/image/begin/01/01-task2-workspace-01.png)
![02-task2-workspace-02](../../../assets/image/begin/01/01-task2-workspace-02.png)

查看conda版本与创建vscode环境以及激活该环境
```shell
conda --version
conda create -n vscode python=3.10
conda activate vscode
```
![02-task2-conda](../../../assets/image/begin/01/01-task2-conda.png)


历史操作
```shell
    1  cd camp3/l0/linux/
    2  python hello_world.py 
    3  conda --version
    4  conda create -n vscode python=3.10
    5  conda actiate vscode
    6  conda activate vscode
    7  history
```

![02-task2-history-operation](../../../assets/image/begin/01/01-task2-history-operation.png)


## 可选任务 3
> 创建并运行`test.sh`文件

导出vscode环境
```shell
conda env export --name vscode > vscode.yml
```
![02-task3-env-export](../../../assets/image/begin/01/01-task3-env-export.png)


移除vscode环境
```shell
conda deactivate
conda remove --name vscode --all
```
![02-task3-env-remove](../../../assets/image/begin/01/01-task3-env-remove.png)


创建test脚本
```shell
touch test.sh
chmod +x test.sh
```

test脚本内容
```shell
#!/bin/bash

# 定义导出环境的函数
export_env() {
    local env_name=$1
    echo "正在导出环境: $env_name"
    # 导出环境到当前目录下的env_name.yml文件
    conda env export -n "$env_name" > "$env_name.yml"
    echo "环境导出完成。"
}

# 定义还原环境的函数
restore_env() {
    local env_name=$1
    echo "正在还原环境: $env_name"
    # 从当前目录下的env_name.yml文件还原环境
    conda env create -n "$env_name" -f "$env_name.yml"
    echo "环境还原完成。"
}

# 检查是否有足够的参数
if [ $# -ne 2 ]; then
    echo "使用方法: $0 <操作> <环境名>"
    echo "操作可以是 'export' 或 'restore'"
    exit 1
fi

# 根据参数执行操作
case "$1" in
    export)
        export_env "$2"
        ;;
    restore)
        restore_env "$2"
        ;;
    *)
        echo "未知操作: $1"
        exit 1
        ;;
esac
```

运行脚本与激活vscode环境
```shell
./test.sh restore vscode
conda activate vscode
```
![02-task3-test-result](../../../assets/image/begin/01/01-task3-test-result.png)

历史操作
```shell
    1  cd camp3/l0/linux/
    2  python hello_world.py 
    3  conda --version
    4  conda create -n vscode python=3.10
    5  conda actiate vscode
    6  conda activate vscode
    7  history
    8  conda env export --name vscode > vscode.yml
    9  conda deactiate
   10  conda env deactiate
   11  conda deactivate
   12  conda remove --name vscode --all
   13  touch test.sh
   14  chmod +x test.sh 
   15  ./test.sh restore vscode
   16  conda activate vscode
   17  history
```

![02-task3-history-operation](../../../assets/image/begin/01/01-task3-history-operation.png)
