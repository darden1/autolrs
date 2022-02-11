import time
import string
import random
import logging
import socket
import socketserver
from pathlib import Path
import torch
import torch.distributed as dist

from .autolrs_server import Controller

def find_free_port(host="localhost", port_range=[20000, 65535+1]):
    for port in range(*port_range):
        try:
            ss = socketserver.TCPServer((host, port), None)
            ss.server_close()
            return port
        except OSError:
            pass
    raise OSError("cannot find free port")

def all_reduce(tensor, reduction='sum'):
    if dist.is_available():
        dist.all_reduce(tensor)
        if reduction == 'mean':
            tensor /= dist.get_world_size()
        dist.barrier()

class AutoLRS():
    def __init__(self, model, optimizer, val_fn, min_lr, max_lr,
                    tau_ini=1000, tau_max=8000, tau_dash_ratio=0.1, k=10,
                    warmup_steps=0, warmup_lr=0, summary_steps=1,
                    listening_host='localhost', listening_port=None):
        
        self._net = model
        self._optimizer = optimizer
        self._val_fn = val_fn
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        assert self.min_lr < self.max_lr , f"max_lr: {self.max_lr} is smaller than min_lr: {self.min_lr}. set max_lr larger than min_lr."

        self.tau_ini = tau_ini
        self.tau_max = tau_max
        self.tau_dash_ratio = tau_dash_ratio
        self.k = k
        self._lr = 0.000001
        self._warmup_steps = warmup_steps
        self._warmup_lr = warmup_lr 
        self._global_step = 0
        self._socket = socket.socket()
        self._started = False
        self._summary_steps = summary_steps
        self._checkpoint_path = './.autolrs/autolrs_ckpt_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))  + '.pth'
        self._listening_host = listening_host
        self._listening_port = listening_port
        self.controller_thead = None
        
        self.valid_state_keys = ['_lr', '_global_step', 'controller_thead']
        self.valid_state_keys_controller_thead = ['exploitation_step', 'lr_steps', 'val_freq', 'lr_to_explore', 
                                                'ring_buffer_len', 'global_step', 'last_total_loss', 'loss', 'lr', 
                                                'lr_counter', 'BO_stage', 'val_stage', 'message', 'loss_after_exploitation', 
                                                'ring_loss_buffer', 'exploitation_flag', 'exploitation_counter', 'opt', 
                                                'x_func_dict', 'x_iters', 'func_val_iters', 'init_loss']
        try:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        except:
            self.global_rank = 0
            self.world_size = 1
       
        if not Path(self._checkpoint_path).parent.exists():
            Path(self._checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # clear temporary pth files
            for _pth in Path(self._checkpoint_path).parent.glob("autolrs_ckpt_*.pth"):
                try:
                    _pth.unlink()
                except:
                    pass

        # Synchronize self._checkpoint dir initialization between ddp processes.
        if self.world_size > 1:
            dist.barrier()

        if self.global_rank==0:
            if self._listening_port is None:
                self._listening_port = find_free_port(host=self._listening_host)

            self.controller_thead = Controller(target=Controller, name="controller_thead", daemon=True, 
                                               host=self._listening_host, port=self._listening_port, min_lr=self.min_lr, max_lr=self.max_lr,
                                               tau_ini=self.tau_ini, tau_max=self.tau_max, tau_dash_ratio=self.tau_dash_ratio, k=self.k)
            self.controller_thead.start()
            self.connect_server()
    
    def connect_server(self):
        logging.info("[AutoLRS]: Try to get connection:  ('{}', {})".format(self._listening_host, self._listening_port))
        self._socket.connect((self._listening_host, self._listening_port))

    def _verbose_operation(self, _op):
        if self._global_step % self._summary_steps == 0:
            logging.info("[AutoLRS at {}] {}".format(self._global_step, _op))

    def save_variables(self):
        """Save model parameters and optimizer states."""
        _start_time = time.time()
        torch.save({
            'model_state_dict': self._net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
            }, self._checkpoint_path)
        logging.info("[AutoLRS] backup variables, elapsed: {}s".format(time.time() - _start_time))

    def restore_variables(self):
        _start_time = time.time()
        checkpoint = torch.load(self._checkpoint_path)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("[AutoLRS] restoring variables, elapsed: {}s".format(time.time() - _start_time))

    def step(self, loss):
        if self._global_step < self._warmup_steps:
            # linear warmup
            if self.world_size==1:
                self._lr = (self._warmup_lr / self._warmup_steps) * (self._global_step + 1)
            else: # ddp with world_size > 1
                _lr_tensor = torch.zeros(1).cuda()
                if self.global_rank==0:
                    # linear warmup
                    _lr_tensor[0] = (self._warmup_lr / self._warmup_steps) * (self._global_step + 1)
                all_reduce(_lr_tensor, reduction='sum') # All values except for global rank 0 are 0, so there is no problem taking SUM.
                self._lr = _lr_tensor.item()

        elif not self._started:
            if self.world_size==1:
                self.save_variables()
                logging.info("[AutoLRS] backup trainable variables to CPU") 
                self._started = True
                self._socket.send(",".join(('startBO', str(loss))).encode("utf-8"))
                self._verbose_operation("Start Bayesian Optimization(BO)")
                data = self._socket.recv(1024).decode("utf-8")
                self._verbose_operation("Received data: " + data)
                try:
                    _lr = (float(data.split(",")[-1]))
                    self._lr = _lr
                except ValueError:
                    pass
            else: # ddp with world_size > 1
                self.save_variables()
                if self.global_rank==0:
                    logging.info("[AutoLRS] backup trainable variables to CPU") 
                self._started = True
                _lr_tensor = torch.zeros(1).cuda()
                if self.global_rank==0:
                    self._socket.send(",".join(('startBO', str(loss))).encode("utf-8"))
                    self._verbose_operation("Start Bayesian Optimization(BO)")
                    data = self._socket.recv(1024).decode("utf-8")
                    self._verbose_operation("Received data: " + data)
                    try:
                        _lr = (float(data.split(",")[-1]))
                        _lr_tensor[0] = _lr
                    except ValueError:
                        pass
                all_reduce(_lr_tensor, reduction='sum')
                _lr = _lr_tensor.item()
                if _lr!=0:
                    self._lr = _lr
        else:
            if self.world_size==1:
                self._socket.send(','.join(('loss', str(loss))).encode('utf-8'))
                data = self._socket.recv(1024).decode("utf-8")
                self._verbose_operation("Received data: " + data)

                if data.startswith("restore"):
                    self.restore_variables()
                    self._lr = (float(data.split(",")[-1]))
                    self._verbose_operation("restore trainable variables")
                elif data.startswith("ckpt"):
                    self.save_variables()
                    self._lr = (float(data.split(",")[-1]))
                    self._verbose_operation("backup trainable variables")
                elif data.startswith('evaluate'):
                    val_loss = self._val_fn()               
                    self._socket.send(",".join(("val_loss", str(val_loss))).encode("utf-8"))
                    data = self._socket.recv(1024).decode("utf-8")
                    self._verbose_operation("Received data: " + data)
                    self._lr = (float(data.split(",")[-1]))
                elif data.startswith('save'):
                    pass
                else:
                    pass
            else: # ddp with world_size > 1
                _state_tensor = torch.zeros(1).cuda()
                if self.global_rank==0:
                    self._socket.send(','.join(('loss', str(loss))).encode('utf-8'))
                    data = self._socket.recv(1024).decode("utf-8")
                    self._verbose_operation("Received data: " + data)

                    if data.startswith("restore"):
                        _state = 0
                    elif data.startswith("ckpt"):
                        _state = 1
                    elif data.startswith('evaluate'):
                        _state = 2
                    elif data.startswith('save'):
                        _state = 3
                    else:
                        _state = 4

                    _state_tensor[0] = _state

                all_reduce(_state_tensor, reduction='sum')
                _state = _state_tensor.item()
                
                if _state==0:
                    self.restore_variables()

                    _lr_tensor = torch.zeros(1).cuda()
                    if self.global_rank==0:
                        _lr_tensor[0] = (float(data.split(",")[-1]))
                    all_reduce(_lr_tensor, reduction='sum')
                    self._lr = _lr_tensor.item()
                    if self.global_rank==0:
                        self._verbose_operation("restore trainable variables")

                elif _state==1:
                    self.save_variables()

                    _lr_tensor = torch.zeros(1).cuda()
                    if self.global_rank==0:
                        _lr_tensor[0] = (float(data.split(",")[-1]))
                    all_reduce(_lr_tensor, reduction='sum')
                    self._lr = _lr_tensor.item()

                    if self.global_rank==0:
                        self._verbose_operation("backup trainable variables")
                elif _state==2:
                    val_loss = self._val_fn()
                    if self.global_rank==0:
                        self._socket.send(",".join(("val_loss", str(val_loss))).encode("utf-8"))
                        data = self._socket.recv(1024).decode("utf-8")
                        self._verbose_operation("Received data: " + data)

                    _lr_tensor = torch.zeros(1).cuda()
                    if self.global_rank==0:
                        _lr_tensor[0] = (float(data.split(",")[-1]))
                    all_reduce(_lr_tensor, reduction='sum')
                    self._lr = _lr_tensor.item()
                else:
                    pass

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._lr
            
        self._global_step += 1

    def state_dict(self):
        _state_dict = {}
        for key, val in self.__dict__.items():
            if key in self.valid_state_keys:
                if key=="controller_thead":
                    if self.controller_thead is None:
                        _state_dict[key] = None
                    else:
                        _ct_dict = {}
                        for _key, _val in val.__dict__.items():
                            if _key in self.valid_state_keys_controller_thead:
                                _ct_dict[_key] = _val
                        _state_dict[key] = _ct_dict
                else:
                    _state_dict[key] = val
        return _state_dict

    def load_state_dict(self, state_dict):
        for key, val in state_dict.items():
            if key in self.valid_state_keys:
                if key=="controller_thead" and self.controller_thead is not None:
                    for _key, _val in val.items():
                        if _key in self.valid_state_keys_controller_thead:
                            self.controller_thead.__dict__[_key] = _val
                else:
                    self.__dict__[key] = val
