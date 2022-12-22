from visualdl import LogWriter
from visualdl.server import app
import os
import torch

def init_visualdl(logdir, model = None, net = None, dummy_input = None):
    if model:
        #检查模型文件是否存在
        if not os.path.exists(model):
            # 如果不存在，且输入了net，则必须输入dummy_input用以生成新的onnx文件
            if net:
                assert isinstance(net, torch.nn.Module)
                assert dummy_input is not None
                torch.onnx.export(net, dummy_input, model)
    
    app.run(logdir,
            model=model,
            host="127.0.0.1",
            port=8040,
            cache_timeout=20,
            language=None,
            public_path=None,
            api_only=False,
            open_browser=False)



class Writer:
    def __init__(self, logdir = None, print_freq = 100):
        self.logdir = logdir
        self.print_freq = print_freq
        if logdir:
            self.logwriter = LogWriter(logdir = logdir)

    def echo(self, tag, value, step):
        if self.logdir:
            self.writer.add_scalar()
        else:
            if step % self.print_freq == 0:
                print()

    def __getattr__(self, __name: str) :
        if hasattr(self.logwriter, __name):
            return getattr(self.logwriter, __name)